import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.3c — resolve with a JOINT-BILATERAL upsample. Full-resolution composite from the
// G-buffer; the sparse probe irradiance is gathered over a 3×3 probe neighbourhood with
// weights = spatial(gaussian) × normal-similarity × planar-depth-similarity, so probes on
// a DIFFERENT surface (across an edge) are rejected → no light leaks / edge roughness, and
// the gather doubles as a light edge-aware smooth. Self-emission is added from the voxels.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient, .y = giStrength, .z = normalSharpness, .w = planeSigma (world).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = probe spacing (px), .y = screen width, .z = screen height.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    albedoTex: new VariableMeta("albedoTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    voxelEmission: new VariableMeta("voxelEmission", VariableKind.Texture, `texture_3d<f32>`, {
      viewDimension: "3d",
      textureSampleType: "float",
    }),
    probeIrr: new VariableMeta("probeIrr", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const BG = vec3<f32>(0.043, 0.051, 0.07);

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
const TEX_COORDS = array<vec2f, 6>(
  vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
  vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  out.texCoord = TEX_COORDS[vertexIndex];
  return out;
}

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// Reconstruct world position at pixel px (integer) from reverse-Z depth.
fn world_at(px: vec2<i32>, screen: vec2<f32>) -> vec3<f32> {
  let d = textureLoad(depthTex, px, 0);
  let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / screen;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d);
  return unproject(ndc);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let px = vec2<i32>(floor(input.position.xy));
  let n = textureLoad(normalTex, px, 0);
  let albedo = textureLoad(albedoTex, px, 0).rgb;
  if (n.a < 0.5) {
    return vec4f(BG, 1.0); // background
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  let spacing = uParams2.x;
  let screen = uParams2.yz;
  let world = world_at(px, screen);

  let probeDims = vec2<i32>(textureDimensions(probeIrr));
  let fp = input.position.xy / spacing - vec2<f32>(0.5); // fractional probe coords
  let baseP = vec2<i32>(floor(fp));
  let normalSharp = uParams.z;
  let planeSigma = max(uParams.w, 1e-3);

  // Joint-bilateral gather over a 3×3 probe neighbourhood.
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 1; j = j + 1) {
    for (var i = -1; i <= 1; i = i + 1) {
      let pc = baseP + vec2<i32>(i, j);
      if (pc.x < 0 || pc.y < 0 || pc.x >= probeDims.x || pc.y >= probeDims.y) { continue; }
      let pdata = textureLoad(probeIrr, pc, 0);
      if (pdata.a < 0.5) { continue; } // invalid probe (off-surface)

      let anchorPx = min((vec2<f32>(pc) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
      let apx = vec2<i32>(anchorPx);
      let pn = textureLoad(normalTex, apx, 0);
      if (pn.a < 0.5) { continue; }
      let Np = normalize(pn.rgb * 2.0 - 1.0);
      let pPos = world_at(apx, screen);

      // weights
      let d2 = fp - vec2<f32>(pc);
      let spatial = exp(-dot(d2, d2) * 0.5);
      let nw = pow(max(0.0, dot(N, Np)), normalSharp);
      let plane = abs(dot(N, pPos - world));
      let dw = exp(-plane / planeSigma);
      let w = spatial * nw * dw;

      acc = acc + pdata.rgb * w;
      wsum = wsum + w;
    }
  }
  let irr = select(vec3<f32>(0.0), acc / wsum, wsum > 0.0);

  // Self-emission from the voxel at this surface point.
  let vc = clamp(
    vec3<i32>(floor((world - uGridOrigin.xyz) / uGridOrigin.w)),
    vec3<i32>(0),
    uGridDims.xyz - vec3<i32>(1),
  );
  let selfEmission = textureLoad(voxelEmission, vc, 0).rgb;

  let lit = albedo * (uParams.x + irr * uParams.y) + selfEmission;
  return vec4f(lit, 1.0);
}
`,
);
