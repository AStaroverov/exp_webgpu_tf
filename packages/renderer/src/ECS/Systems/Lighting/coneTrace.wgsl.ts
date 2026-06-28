import { wgsl } from "../../../WGSL/wgsl.ts";

// Shared cone-tracing helpers used by BOTH the per-pixel cone gather (voxelCone.shader.ts)
// and the low-res SH irradiance probe pass (voxelProbe.shader.ts). They read uGridOrigin
// (.xyz world min, .w cellSize) + uGridDims (.xyz voxel counts) + voxelRadiance + voxelSampler
// by GLOBAL name — any shader that inlines this fragment MUST declare those four with
// identical names + types (uGridOrigin: vec4<f32>, uGridDims: vec4<i32>, voxelRadiance:
// texture_3d<f32>, voxelSampler: sampler).
//
// Extracted VERBATIM from the inline trace_cone + build_basis in voxelCone.shader.ts — the
// emitted WGSL is byte-identical, so the per-pixel pass behaves unchanged. The wgsl`` tag
// inlines a no-name fragment's body in-place. NO backtick character may appear inside the
// WGSL comments below.
export const coneTrace = wgsl /* wgsl */ `
// Orthonormal basis with column 2 = n (so basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// One cone marched along dir from origin: diameter grows with distance, each step samples
// voxelRadiance at LOD=log2(diameter/voxelSize) and composites front-to-back ("over").
// reach = how far this cone marches (world units). The step is floored at reach/maxSteps so the
// cone ALWAYS spans its full reach within the step budget — without this, a narrow cone's
// tiny near-field steps burn the budget and it dies ~16 units out (the "light stops at cone
// reach" problem). fadeFrac>0 tapers the gathered radiance over the last fadeFrac of reach
// (hides the artificial cutoff of the fill cones); pass 0 for aimed cones, which end at the
// real light, so they must NOT be tapered. Occlusion (alpha) is never tapered -> shadows stay.
// maxSteps = march budget (aimed cones want it high for sharp shadows; the low-frequency fill
// hemisphere is fine with ~half — its bounce is smooth, so fewer steps barely change it but
// halve the dominant cone cost). alphaCut = early-out opacity: a fill cone that is ~opaque adds
// almost nothing more, so cutting at <1 saves the tail steps; aimed cones pass 1.0 (no early cut).
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, aperture: f32, reach: f32, fadeFrac: f32, startJ: f32, maxSteps: i32, alphaCut: f32) -> vec4<f32> {
  var col = vec3<f32>(0.0);
  var alpha = 0.0;
  let voxelSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let extent = vec3<f32>(uGridDims.xyz) * voxelSize;
  let stepFloor = reach / f32(maxSteps);
  // Per-pixel SCALE of the start distance (startJ in [0,1)). Because the march is ~geometric
  // (step grows with distance), scaling dist0 shifts the sampling "shells" by a per-pixel
  // factor at EVERY distance -> the concentric "tree-ring" banding around bright sources
  // dithers across pixels and is blurred away by the half-res upsample. Lets few cones look
  // smooth without going to 48.
  var dist = voxelSize * (1.0 + startJ);
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (alpha >= alphaCut || dist > reach) { break; }
    let diameter = max(voxelSize, 2.0 * aperture * dist);
    let lod = log2(diameter / voxelSize);
    let wp = origin + dir * dist;
    let uvw = (wp - gridMin) / extent;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s = textureSampleLevel(voxelRadiance, voxelSampler, uvw, lod);
    var window = 1.0;
    if (fadeFrac > 0.0) {
      window = clamp((reach - dist) / (reach * fadeFrac), 0.0, 1.0);
    }
    col = col + (1.0 - alpha) * s.rgb * window;
    alpha = alpha + (1.0 - alpha) * s.a;
    dist = dist + max(diameter * 0.5, stepFloor);
  }
  return vec4<f32>(col, alpha);
}
`;
