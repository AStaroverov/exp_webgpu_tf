import { wgsl } from "../../../WGSL/wgsl.ts";

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
