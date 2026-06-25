import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { voxelTrace } from "./voxelTrace.wgsl.ts";

// Base directions per side for cascade 0 (octahedral DIR_W0×DIR_W0 over the sphere).
// Cascade c has DIR_W0*2^c directions per side and probes/2^c per side, so every cascade
// atlas is the SAME size: (probesX0*DIR_W0) × (probesY0*DIR_W0). Small base = cheap near
// field; the merge brings far-field angular detail.
export const CASCADE_DIR_W = 4;

// Stage 2.3b — directional probe gather, per cascade. Renders ONE cascade's atlas: one
// texel per (probe, direction). The cascade's spacing / dirs-per-side / interval come from
// the per-cascade uniform uCascade (a distinct buffer per cascade — see createVoxelSystem,
// which avoids the writeBuffer-overwrite hazard). Each direction is voxel-DDA-traced over
// the cascade interval; the texel stores rgb = interval radiance, a = visibility.

const sampled3d = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    viewDimension: "3d",
    textureSampleType: "float",
  });

export const shaderMeta = new ShaderMeta(
  {
    // .x = normalBias, .y = sky intensity (top-cascade miss).
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = screen width, .y = screen height.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    // Per-cascade: [0] = (spacing, dirsPerSide, intervalStart, intervalEnd), [1].x = isTop.
    cascade: new VariableMeta("uCascade", VariableKind.Uniform, `array<vec4<f32>, 2>`),
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
  let spacing = uCascade[0].x;
  let dirsSide = i32(uCascade[0].y);
  let intStart = uCascade[0].z;
  let intEnd = uCascade[0].w;
  let isTop = uCascade[1].x > 0.5;

  // Atlas texel -> (probe, direction) for THIS cascade's dirsSide.
  let texel = vec2<i32>(floor(input.position.xy));
  let probeCoord = texel / dirsSide;
  let dirCoord = texel - probeCoord * dirsSide;

  let screen = uParams2.xy;
  let anchorPx = min((vec2<f32>(probeCoord) + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
  let px = vec2<i32>(anchorPx);

  let n = textureLoad(normalTex, px, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 0.0); // no surface under this probe
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  let depth = textureLoad(depthTex, px, 0);
  let uv = anchorPx / screen;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let worldPos = unproject(ndc);

  let cellSize = uGridOrigin.w;
  let e = (vec2<f32>(dirCoord) + vec2<f32>(0.5)) / f32(dirsSide) * 2.0 - 1.0;
  let dir = oct_decode(e);
  let origin = worldPos + N * (cellSize * 1.5 + uParams.x);

  return trace_interval(origin, dir, intStart, intEnd, isTop, uParams.y);
}
`,
);
