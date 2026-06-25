import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { voxelTrace } from "./voxelTrace.wgsl.ts";

// Cascade-0 directions per side (octahedral DIR_W×DIR_W tile per probe → DIR_W² directions
// over the sphere). The probe atlas has size (probesX*DIR_W)×(probesY*DIR_W); cascade c
// (Stage 2.3b) keeps that size constant (probes/2^c per side × DIR_W*2^c dirs per side).
export const CASCADE_DIR_W = 8;

// Stage 2.3a — directional probe gather. Renders the probe ATLAS: one texel per
// (probe, direction). Each probe is placed on the visible surface from the G-buffer
// (anchor pixel → depth/normal), and each direction is a DETERMINISTIC octahedral ray
// voxel-DDA-traced over the cascade interval. The atlas texel stores rgb = directional
// radiance, a = validity (0 = no surface under the probe). The hemisphere integration to
// irradiance happens in the separate integrate pass.

const sampled3d = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    viewDimension: "3d",
    textureSampleType: "float",
  });

export const shaderMeta = new ShaderMeta(
  {
    // .x = probe spacing (px), .y = maxDist (interval length), .z = normalBias, .w = sky.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width, .y = screen height, .z = interval start (world).
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    voxelAlbedo: sampled3d("voxelAlbedo"),
    voxelEmission: sampled3d("voxelEmission"),
    depthTex: new VariableMeta("depthTex", VariableKind.Texture, `texture_depth_2d`, {
      textureSampleType: "depth",
    }),
    normalTex: new VariableMeta("normalTex", VariableKind.Texture, `texture_2d<f32>`, {
      textureSampleType: "float",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const DIR_W: i32 = ${CASCADE_DIR_W};

const POSITION = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VertexOutput { @builtin(position) position: vec4f };

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  return out;
}

${voxelTrace}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // Atlas texel -> (probe, direction): probe = texel / DIR_W, dir = texel % DIR_W.
  let texel = vec2<i32>(floor(input.position.xy));
  let probeCoord = texel / DIR_W;
  let dirCoord = texel - probeCoord * DIR_W;

  let spacing = uParams.x;
  let screen = uParams2.xy;
  let anchorPx = min((vec2<f32>(probeCoord) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
  let px = vec2<i32>(anchorPx);

  let n = textureLoad(normalTex, px, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  let depth = textureLoad(depthTex, px, 0);
  let uv = anchorPx / screen;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let worldPos = unproject(ndc);

  let cellSize = uGridOrigin.w;
  // Octahedral direction for this atlas cell (full sphere; cosine weighting is applied
  // later in the integrate pass).
  let e = (vec2<f32>(dirCoord) + vec2<f32>(0.5)) / f32(DIR_W) * 2.0 - 1.0;
  let dir = oct_decode(e);

  // Trace the cascade interval [intervalStart, intervalStart + maxDist] along dir.
  let intervalStart = uParams2.z;
  let origin = worldPos + N * (cellSize * 1.5 + uParams.z) + dir * intervalStart;
  let radiance = trace_radiance(origin, dir, uParams.y, uParams.w);

  return vec4f(radiance, 1.0);
}
`,
);
