import { wgsl } from "../../../../../WGSL/wgsl.ts";

// SHARED WGSL fragments for the screen-probe passes, inlined via the wgsl-template interpolation
// (the old coneTrace.wgsl.ts pattern): a fragment with no `name` has its body spliced in verbatim
// wherever it is interpolated, so there is ONE source of truth and no risk of copies drifting.

// The bilateral plane-normal weight: how well a candidate probe (Pp, Np) can represent a surface
// point (P, N). planeDist = the probe's signed distance out of P's tangent plane (rejects probes
// across a depth discontinuity); the normal term rejects probes on a differently-oriented face.
// Shared by the resolve's tap weighting AND the gather's temporal history validation — same
// semantics, one source of truth.
export const probeWeightWGSL = wgsl /* wgsl */ `
fn sp_plane_normal_weight(P: vec3<f32>, N: vec3<f32>, Pp: vec3<f32>, Np: vec3<f32>, planeThresh: f32, normalPow: f32) -> f32 {
  let planeDist = abs(dot(Pp - P, N));
  let wp = clamp(1.0 - planeDist / planeThresh, 0.0, 1.0);
  let wn = pow(max(dot(Np, N), 0.0), normalPow);
  return wp * wn;
}
`;
