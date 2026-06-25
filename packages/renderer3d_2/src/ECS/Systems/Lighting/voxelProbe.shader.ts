import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { voxelTrace } from "./voxelTrace.wgsl.ts";

// Stage 2.2 (G-buffer primary) — probe gather. Renders the SPARSE screen-space probe grid
// (one texel per probe). Each probe is placed on the visible surface read from the
// G-buffer at the centre of its screen cell (depth → world pos, normal), then gathers the
// hemisphere (K cosine rays) by voxel-DDA tracing → irradiance. The primary visibility is
// the cheap rasterized SDF G-buffer (no full-res voxel DDA). Output texel =
// vec4(irradiance.rgb, valid).

const sampled3d = (name: string) =>
  new VariableMeta(name, VariableKind.Texture, `texture_3d<f32>`, {
    viewDimension: "3d",
    textureSampleType: "float",
  });

export const shaderMeta = new ShaderMeta(
  {
    // .x = numRays, .y = maxDist (world), .z = normalBias, .w = skyIntensity.
    params: new VariableMeta("uParams", VariableKind.Uniform, `vec4<f32>`),
    // .x = probe spacing (px), .y = screen width, .z = screen height, .w = seed/frame.
    params2: new VariableMeta("uParams2", VariableKind.Uniform, `vec4<f32>`),
    invViewProj: new VariableMeta("uInvViewProj", VariableKind.Uniform, `mat4x4<f32>`),
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    gridDims: new VariableMeta("uGridDims", VariableKind.Uniform, `vec4<i32>`),
    voxelAlbedo: sampled3d("voxelAlbedo"),
    voxelEmission: sampled3d("voxelEmission"),
    // G-buffer (full canvas res): reverse-Z depth + packed world normal (a = surface mask).
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
  // This fragment IS one probe; its target texel = the probe's grid coord.
  let probeCoord = floor(input.position.xy);
  let spacing = uParams2.x;
  let screen = uParams2.yz;

  // Anchor = centre of this probe's screen cell; clamp to the framebuffer.
  let anchorPx = min((probeCoord + vec2<f32>(0.5)) * spacing, screen - vec2<f32>(1.0));
  let px = vec2<i32>(anchorPx);

  let n = textureLoad(normalTex, px, 0);
  if (n.a < 0.5) {
    return vec4f(0.0, 0.0, 0.0, 0.0); // no surface under this probe
  }
  let N = normalize(n.rgb * 2.0 - 1.0);

  // Reconstruct the probe's world position from reverse-Z depth.
  let depth = textureLoad(depthTex, px, 0);
  let uv = anchorPx / screen;
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth);
  let worldPos = unproject(ndc);

  let cellSize = uGridOrigin.w;
  let origin = worldPos + N * (cellSize * 1.5 + uParams.z);

  let maxDist = uParams.y;
  let sky = uParams.w;

  // DETERMINISTIC directions: a D×D octahedral grid over the full sphere, cosine-weighted
  // by the surface normal (below-horizon dirs drop out). No randomness → no Monte-Carlo
  // grain; only smooth angular discretization. D ≈ sqrt(numRays).
  let D = clamp(i32(round(sqrt(uParams.x))), 2, 16);
  var acc = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var v = 0; v < D; v = v + 1) {
    for (var u = 0; u < D; u = u + 1) {
      let e = (vec2<f32>(f32(u), f32(v)) + 0.5) / f32(D) * 2.0 - 1.0;
      let dir = oct_decode(e);
      let w = max(0.0, dot(dir, N));
      if (w > 0.0) {
        acc = acc + trace_radiance(origin, dir, maxDist, sky) * w;
        wsum = wsum + w;
      }
    }
  }
  let irr = acc / max(wsum, 1e-4);

  return vec4f(irr, 1.0);
}
`,
);
