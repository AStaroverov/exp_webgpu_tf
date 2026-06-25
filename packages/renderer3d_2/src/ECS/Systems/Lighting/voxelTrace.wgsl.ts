import { wgsl } from "../../../WGSL/wgsl.ts";

// Shared voxel-tracing helpers used by the GI / probe / resolve passes. Inlined verbatim
// (no `name` → the wgsl tag splices the body in place). Any shader that inlines this MUST
// declare these bindings by global name with identical types:
//   uInvViewProj : mat4x4<f32>      (inverse reverse-Z viewProj, for unproject)
//   uGridOrigin  : vec4<f32>        (.xyz world min corner, .w cellSize)
//   uGridDims    : vec4<i32>        (.xyz voxel counts per axis)
//   voxelAlbedo  : texture_3d<f32>  (rgb albedo, a = occupancy)
//   voxelEmission: texture_3d<f32>  (rgb emission)
export const voxelTrace = wgsl /* WGSL */ `
const PI: f32 = 3.14159265;

// Unproject an NDC point (z reverse-Z) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
  let w = uInvViewProj * vec4<f32>(ndc, 1.0);
  return w.xyz / w.w;
}

// PCG hash (uniform u32 stream).
fn pcg(v_in: u32) -> u32 {
  var v = v_in * 747796405u + 2891336453u;
  let s = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (s >> 22u) ^ s;
}

// Octahedral decode: maps e in [-1,1]^2 to a unit direction on the sphere. Used to
// enumerate a DETERMINISTIC, evenly-spread set of directions (no Monte-Carlo noise).
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    let s = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
    v = vec3<f32>((1.0 - abs(v.yx)) * s, v.z);
  }
  return normalize(v);
}

// Orthonormal basis with column 2 = n (basis * (x,y,z) = x*t + y*b + z*n).
fn build_basis(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  let b = cross(n, t);
  return mat3x3<f32>(t, b, n);
}

// DDA (Amanatides–Woo) to the first solid voxel (occupancy a>0.5) within maxT world
// units. hit=false on a miss / grid exit / out-of-range.
struct VHit { hit: bool, coord: vec3<i32>, normal: vec3<f32> };

fn dda(ro: vec3<f32>, rd: vec3<f32>, maxT: f32) -> VHit {
  var res: VHit;
  res.hit = false;
  res.coord = vec3<i32>(0);
  res.normal = vec3<f32>(0.0, 0.0, 1.0);

  let cellSize = uGridOrigin.w;
  let gridMin = uGridOrigin.xyz;
  let dims = uGridDims.xyz;
  let gridMax = gridMin + vec3<f32>(dims) * cellSize;

  let inv = 1.0 / rd;
  let ta = (gridMin - ro) * inv;
  let tb = (gridMax - ro) * inv;
  let tsmall = min(ta, tb);
  let tbig = max(ta, tb);
  let tEnter = max(max(tsmall.x, tsmall.y), tsmall.z);
  let tExit = min(min(tbig.x, tbig.y), tbig.z);
  if (tExit < max(tEnter, 0.0)) { return res; }

  let tStart = max(tEnter, 0.0);
  if (tStart > maxT) { return res; }

  let pEnter = ro + rd * (tStart + cellSize * 1e-3);
  var coord = clamp(
    vec3<i32>(floor((pEnter - gridMin) / cellSize)),
    vec3<i32>(0),
    dims - vec3<i32>(1),
  );

  let step = vec3<i32>(sign(rd));
  let tDelta = abs(cellSize * inv);
  let nextBoundary = gridMin + (vec3<f32>(coord) + max(vec3<f32>(step), vec3<f32>(0.0))) * cellSize;
  var tMax = (nextBoundary - ro) * inv;

  var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (tsmall.x >= tsmall.y && tsmall.x >= tsmall.z) {
    normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
  } else if (tsmall.y >= tsmall.z) {
    normal = vec3<f32>(0.0, -f32(step.y), 0.0);
  } else {
    normal = vec3<f32>(0.0, 0.0, -f32(step.z));
  }

  let maxSteps = dims.x + dims.y + dims.z;
  var tCur = tStart;
  for (var i = 0; i < maxSteps; i = i + 1) {
    if (textureLoad(voxelAlbedo, coord, 0).a > 0.5) {
      res.hit = true;
      res.coord = coord;
      res.normal = normal;
      return res;
    }
    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
      tCur = tMax.x;
      coord.x = coord.x + step.x;
      tMax.x = tMax.x + tDelta.x;
      normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
      if (coord.x < 0 || coord.x >= dims.x) { return res; }
    } else if (tMax.y <= tMax.z) {
      tCur = tMax.y;
      coord.y = coord.y + step.y;
      tMax.y = tMax.y + tDelta.y;
      normal = vec3<f32>(0.0, -f32(step.y), 0.0);
      if (coord.y < 0 || coord.y >= dims.y) { return res; }
    } else {
      tCur = tMax.z;
      coord.z = coord.z + step.z;
      tMax.z = tMax.z + tDelta.z;
      normal = vec3<f32>(0.0, 0.0, -f32(step.z));
      if (coord.z < 0 || coord.z >= dims.z) { return res; }
    }
    if (tCur > maxT) { return res; }
  }
  return res;
}

// Incoming radiance along a secondary ray: emission of the first solid voxel hit, or the
// sky term on a miss within maxDist.
fn trace_radiance(origin: vec3<f32>, dir: vec3<f32>, maxDist: f32, sky: f32) -> vec3<f32> {
  let h = dda(origin, dir, maxDist);
  if (h.hit) {
    return textureLoad(voxelEmission, h.coord, 0).rgb;
  }
  return vec3<f32>(sky);
}

// Radiance Cascades interval trace over the shell [tStart, tEnd] along dir. Returns
// rgb = radiance gathered in the interval, a = visibility (1 = the interval was clear, so
// the FARTHER cascade may contribute; 0 = an occluder was hit, blocking the far light).
//   - hit an opaque voxel in the interval → (its emission, 0)
//   - reached the interval end clear → (0, 1), or (sky, 1) for the top cascade
fn trace_interval(
  origin: vec3<f32>, dir: vec3<f32>, tStart: f32, tEnd: f32, isTop: bool, sky: f32,
) -> vec4<f32> {
  let h = dda(origin + dir * tStart, dir, tEnd - tStart);
  if (h.hit) {
    return vec4<f32>(textureLoad(voxelEmission, h.coord, 0).rgb, 0.0);
  }
  if (isTop) {
    return vec4<f32>(vec3<f32>(sky), 1.0);
  }
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`;
