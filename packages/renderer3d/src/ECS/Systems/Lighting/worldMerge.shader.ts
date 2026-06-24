import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";

// Stage-2 world-space Radiance Cascades — MERGE pass (doc §5).
//
// Runs once per cascade, top-down (c = N-2 .. 0). Fullscreen over the destination
// cascade's atlas (probeMerge[c]). For each (probe_c, dir_c) cell it merges the
// already-merged COARSER cascade c+1 into this cascade's raw gather:
//
//   1. near  = probeRad[c] at this cell (the gather of THIS cascade).
//   2. P     = world XY of dst probe (i,j).
//   3. In the c+1 grid (2x cell spacing) find the 4 probes whose cell contains P
//      and their BILINEAR weights — this replaces src-dgi's hash-grid "4 nearest
//      surfels" with plain regular-grid interpolation (doc §5).
//   4. dir_c maps to a 2x2 block of directions in c+1's finer tile (angular x4):
//      srcDirBase = uv*2; average the 4 sub-intervals.
//   5. far   = that averaged coarse interval; merged = merge_intervals(near, far);
//      accumulate merged * bilinearWeight over the 4 neighbors, normalize.
//
// near = probeRad[c], far source = probeMerge[c+1] (or probeRad[N-1] for the top
// cascade, which has no merge target). Output -> probeMerge[c].
//
// COORDINATES: world Z-up; merge works in world XY + tile index space (no octahedral
// decode needed — directions map by 2x2 index blocks).

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : textures ----
    // THIS cascade's raw gather (the "near" interval). 2D-array: one slice per z layer.
    nearTexture: new VariableMeta("nearTexture", VariableKind.Texture, `texture_2d_array<f32>`, {
      viewDimension: "2d-array",
    }),
    // The coarser cascade's already-merged radiance (the "far" interval source).
    srcTexture: new VariableMeta("srcTexture", VariableKind.Texture, `texture_2d_array<f32>`, {
      viewDimension: "2d-array",
    }),

    // ---- group 0 : uniforms (dst = this cascade, src = c+1) ----
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    gridOriginSrc: new VariableMeta("uGridOriginSrc", VariableKind.Uniform, `vec4<f32>`),
    gridX: new VariableMeta("uGridX", VariableKind.Uniform, `u32`),
    gridY: new VariableMeta("uGridY", VariableKind.Uniform, `u32`),
    dirW: new VariableMeta("uDirW", VariableKind.Uniform, `u32`),
    cell: new VariableMeta("uCell", VariableKind.Uniform, `f32`),
    gridXSrc: new VariableMeta("uGridXSrc", VariableKind.Uniform, `u32`),
    gridYSrc: new VariableMeta("uGridYSrc", VariableKind.Uniform, `u32`),
    dirWSrc: new VariableMeta("uDirWSrc", VariableKind.Uniform, `u32`),
    cellSrc: new VariableMeta("uCellSrc", VariableKind.Uniform, `f32`),
    // Array layer (z slice) this pass renders / reads. Layers are independent.
    layer: new VariableMeta("uLayer", VariableKind.Uniform, `u32`),
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
  var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  return output;
}

// near over far with the visibility term (verbatim from src-dgi cascade.slang):
// far radiance only carries through if the near interval was unobstructed (near.a).
fn merge_intervals(near: vec4<f32>, far: vec4<f32>) -> vec4<f32> {
  let radiance = near.rgb + near.a * far.rgb;
  let visibility = near.a * far.a;
  return vec4<f32>(radiance, visibility);
}

// World XY of dst probe (i,j) — must match worldGather.probe_world_xy for cascade c.
fn dst_probe_world_xy(ij: vec2<u32>) -> vec2<f32> {
  let center = vec2<f32>(f32(uGridX), f32(uGridY)) * 0.5;
  return uGridOrigin.xy + (vec2<f32>(ij) + 0.5 - center) * uCell;
}

// Continuous probe-grid coordinate of a world XY in the SRC (c+1) grid.
fn src_coord(wxy: vec2<f32>) -> vec2<f32> {
  let center = vec2<f32>(f32(uGridXSrc), f32(uGridYSrc)) * 0.5;
  return (wxy - uGridOriginSrc.xy) / uCellSrc + center - 0.5;
}

const OFFSETS = array<vec2<i32>, 4>(
  vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(1, 1)
);

// Average the 2x2 far sub-interval block of src probe (ij_n) for dst dir cell uv,
// merged over the near interval.
fn merge_neighbor(near: vec4<f32>, ij_n: vec2<i32>, uv: vec2<u32>) -> vec4<f32> {
  let tileOrigin = ij_n * i32(uDirWSrc);
  let srcDirBase = vec2<i32>(uv) * 2;
  var far = vec4<f32>(0.0);
  for (var k = 0; k < 4; k = k + 1) {
    far = far + textureLoad(srcTexture, tileOrigin + srcDirBase + OFFSETS[k], i32(uLayer), 0) * 0.25;
  }
  return merge_intervals(near, far);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  let px = vec2<u32>(floor(input.position.xy));
  let ij = px / uDirW;          // dst probe index
  let uv = px % uDirW;          // dst direction cell

  let near = textureLoad(nearTexture, vec2<i32>(px), i32(uLayer), 0);
  let P = dst_probe_world_xy(ij);

  // 4 bilinear neighbors in the coarse (c+1) grid.
  let gc = src_coord(P);
  let g0 = floor(gc);
  let f = gc - g0;
  let i0 = vec2<i32>(g0);
  let maxIdx = vec2<i32>(i32(uGridXSrc) - 1, i32(uGridYSrc) - 1);
  let n00 = clamp(i0 + vec2<i32>(0, 0), vec2<i32>(0), maxIdx);
  let n10 = clamp(i0 + vec2<i32>(1, 0), vec2<i32>(0), maxIdx);
  let n01 = clamp(i0 + vec2<i32>(0, 1), vec2<i32>(0), maxIdx);
  let n11 = clamp(i0 + vec2<i32>(1, 1), vec2<i32>(0), maxIdx);
  let w00 = (1.0 - f.x) * (1.0 - f.y);
  let w10 = f.x * (1.0 - f.y);
  let w01 = (1.0 - f.x) * f.y;
  let w11 = f.x * f.y;

  let acc = merge_neighbor(near, n00, uv) * w00
          + merge_neighbor(near, n10, uv) * w10
          + merge_neighbor(near, n01, uv) * w01
          + merge_neighbor(near, n11, uv) * w11;
  // Weights already sum to 1; emit the merged interval for this cascade.
  return acc;
}
    `,
);
