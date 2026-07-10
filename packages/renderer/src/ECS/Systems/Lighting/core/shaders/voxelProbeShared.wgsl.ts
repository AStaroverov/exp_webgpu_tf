import { wgsl } from "../../../../../WGSL/wgsl.ts";

// SHARED WGSL fragments for the adaptive screen-probe atlas (Phase 1). These are inlined into
// MULTIPLE probe shaders via the wgsl-template interpolation (the old coneTrace.wgsl.ts pattern):
// a fragment with no `name` has its body spliced in verbatim wherever it is interpolated, so there
// is ONE source of truth and no risk of two copies drifting apart. The whole point of factoring
// these out is the CRITICAL consistency rule: the plane-normal weight that DECIDES where an
// adaptive probe is placed (voxelProbeRefine) MUST be byte-identical to the one the resolve uses to
// SELECT probes (voxelCone.resolve_screen_probes) — otherwise we place probes the resolve won't
// pick (wasted trace) or reject pixels we did refine (the artifact we are fixing).

// probeData record pack/unpack. One vec4<u32> per probe (16B, keeps std430 alignment): .x = the
// representative full-res pixel packed as (px.x | px.y<<16), .y = subdivision level (0 = 16px
// uniform, 1 = 8px adaptive), .z = the probe FOOTPRINT (its cell size in full-res px — the coarse
// tile for a uniform probe, cellPx for an adaptive one — set by classify/refine after packing; it
// area-weights the probe in the resolve for a density-invariant average), .w spare. sp_pack_probe
// leaves .z = 0; the placement pass fills it. The representative pixel is all the gather/resolve need
// — P and N are reconstructed from the G-buffer at that pixel exactly as the flat-atlas build does.
export const probePackWGSL = wgsl /* wgsl */ `
fn sp_pack_probe(px: vec2<i32>, level: u32) -> vec4<u32> {
  return vec4<u32>((u32(px.x) & 0xffffu) | ((u32(px.y) & 0xffffu) << 16u), level, 0u, 0u);
}
fn sp_unpack_pixel(rec: vec4<u32>) -> vec2<i32> {
  return vec2<i32>(i32(rec.x & 0xffffu), i32((rec.x >> 16u) & 0xffffu));
}
`;

// FOVEATED RING density levels. The uniform probe lattice thins out with radial distance from the
// SCREEN CENTER (the camera's point of interest): level 0 = one probe per tile (the flat grid),
// level 1 = one probe per 2×2-tile block, level 2 = one per 4×4 block — probe count ÷4 per level.
// The level function is shared by classify (placement), decide (trigger stride), refine (gating),
// the cone resolve (window stride) and the debug view — ONE definition, or the passes disagree on
// which tiles hold probes and the resolve taps empty texels.
//
// BOOST: the SAME per-block hysteresis state that drives fine refinement in the center restores one
// density step in a sparse ring (level 2 → 1, 1 → 0) — "уплотнение тем же множителем": a lighting
// gradient at the screen edge gets its density back, a flat dark ring stays at ÷16. The state is
// owned by the block's BASE anchor tile (top-left of the base-level block), so ANY tile can find it:
// base level from its own radius → anchor → refineState[anchor]. Anchors NEST (a multiple of 4 is
// also a multiple of 2), so a stride-s window in a coarse ring always lands on valid probe positions
// even where a neighboring region is finer.
export const probeRingWGSL = wgsl /* wgsl */ `
// A block is refinement-ACTIVE while its decide hysteresis counter >= this (see voxelProbeDecide).
const SP_RING_HYST_ACTIVE: u32 = 8u;

// Base density level (0/1/2) of a coarse tile from its radial screen-center distance, normalized so
// the screen CORNER = 1. r0/r1 are the ring thresholds (>= ~8 = rings off, level 0 everywhere).
fn sp_ring_base_level(tc: vec2<i32>, screenPx: vec2<f32>, tilePx: f32, r0: f32, r1: f32) -> i32 {
  let c = (vec2<f32>(tc) + vec2<f32>(0.5)) * tilePx;
  let r = length(c - screenPx * 0.5) / max(0.5 * length(screenPx), 1e-4);
  if (r >= r1) { return 2; }
  if (r >= r0) { return 1; }
  return 0;
}

// Anchor (top-left tile) of the 2^level-tile block containing tc. -(1<<level) == ~(blockSize-1).
fn sp_ring_anchor(tc: vec2<i32>, level: i32) -> vec2<i32> {
  return tc & vec2<i32>(-(1 << u32(level)));
}
`;

// The bilateral plane-normal weight: how well a candidate probe (Pp, Np) can represent a surface
// point (P, N). planeDist = the probe's signed distance out of P's tangent plane (rejects probes
// across a depth discontinuity); the normal term rejects probes on a differently-oriented face.
// This is EXACTLY the wp*wn product the resolve multiplies onto its bilinear/spatial weight — the
// refine pass sums it over the uniform cage and spawns a probe only where the sum is ~0 (no uniform
// probe can represent the cell), which is precisely where the resolve would fall back to the loose
// blend. Byte-identical placement/resolve weight = no wasted probes, no persistent artifacts.
export const probeWeightWGSL = wgsl /* wgsl */ `
fn sp_plane_normal_weight(P: vec3<f32>, N: vec3<f32>, Pp: vec3<f32>, Np: vec3<f32>, planeThresh: f32, normalPow: f32) -> f32 {
  let planeDist = abs(dot(Pp - P, N));
  let wp = clamp(1.0 - planeDist / planeThresh, 0.0, 1.0);
  let wn = pow(max(dot(Np, N), 0.0), normalPow);
  return wp * wn;
}
`;
