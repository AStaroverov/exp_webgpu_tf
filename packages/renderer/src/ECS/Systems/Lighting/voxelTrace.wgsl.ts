import { wgsl } from "../../../WGSL/wgsl.ts";

// SHARED WGSL fragments used by more than one lighting shader, inlined via wgsl-template
// interpolation (same pattern as voxelProbeShared.wgsl.ts): a fragment with no `name` is spliced in
// verbatim, so there is ONE source of truth and the copies can't drift.
//
// These cover ONLY the geometry helpers that are genuinely identical across shaders. The cone-march
// itself is NOT shared: voxelScreenProbe's trace samples through sample_radiance (iso/aniso toggle,
// needs the 6 directional volumes + uLightParams) while voxelCone's short AO trace reads the plain
// isotropic voxelRadiance pyramid — the cone shader deliberately does not bind the aniso volumes.

// Unproject an NDC point (z reverse-Z) to world space. The inverse-viewProj is passed in so every
// caller can use its own uniform (uInvViewProj in cone/screen-probe, uF.invViewProj in composite).
export const unprojectWGSL = wgsl /* wgsl */ `
fn unproject(ndc: vec3<f32>, invVP: mat4x4<f32>) -> vec3<f32> {
  let w = invVP * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}
`;

// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
export const buildBasisWGSL = wgsl /* wgsl */ `
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}
`;
