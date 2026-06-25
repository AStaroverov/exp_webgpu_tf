import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage 2.2a (G-buffer primary) — resolve. Full-resolution composite from the G-buffer:
// per pixel reads albedo (renderTexture) + surface mask (normal.a), reconstructs the world
// position from depth to fetch the voxel self-emission, and bilinearly samples the sparse
// probe irradiance:  lit = albedo*(ambient + irr*strength) + selfEmission.
// No voxel DDA here (cheap full-res). No geometric weighting yet (2.2b) → edge leaks.

export const shaderMeta = new ShaderMeta(
  {
    // .x = ambient, .y = giStrength.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = probe spacing (px).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    // .xyz = world min corner, .w = cellSize.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // .xyz = voxel counts per axis.
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
    probeSampler: new VariableMeta("probeSampler", VariableKind.Sampler, `sampler`),
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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let px = vec2<i32>(floor(input.position.xy));
  let n = textureLoad(normalTex, px, 0);
  let albedo = textureLoad(albedoTex, px, 0).rgb;
  if (n.a < 0.5) {
    return vec4f(BG, 1.0); // background
  }

  // Self-emission: reconstruct the world position from reverse-Z depth, look up the voxel.
  let depth = textureLoad(depthTex, px, 0);
  let ndc = vec3<f32>(input.texCoord.x * 2.0 - 1.0, (1.0 - input.texCoord.y) * 2.0 - 1.0, depth);
  let world = unproject(ndc);
  let vc = clamp(
    vec3<i32>(floor((world - uGridOrigin.xyz) / uGridOrigin.w)),
    vec3<i32>(0),
    uGridDims.xyz - vec3<i32>(1),
  );
  let selfEmission = textureLoad(voxelEmission, vc, 0).rgb;

  // Sample the sparse probe irradiance. probe texel pc centre maps to screen pixel
  // (pc+0.5)*spacing, so probeUv = fragCoord / (spacing * probeDims) aligns exactly.
  let probeDims = vec2<f32>(textureDimensions(probeIrr));
  let probeUv = input.position.xy / (uParams2.x * probeDims);
  let irr = textureSampleLevel(probeIrr, probeSampler, probeUv, 0.0).rgb;

  let lit = albedo * (uParams.x + irr * uParams.y) + selfEmission;
  return vec4f(lit, 1.0);
}
`,
);
