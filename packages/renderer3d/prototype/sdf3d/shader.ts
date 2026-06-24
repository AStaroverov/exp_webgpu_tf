// WGSL for the 2.5D SDF impostor prototype.
//
// One draw call, instanced. Per instance we rasterize a world-space bounding
// box (the "impostor") and, in the fragment shader, sphere-trace the instance's
// local 3D SDF along the camera ray. The ray hit gives a real world-space point
// → we write its clip depth to frag_depth, so the depth buffer sorts/occludes
// everything correctly with zero CPU sorting. A single directional light shades
// the surface normal so volumes read as plausibly 3D.
//
// Stage 1: the full game 2D-shape vocabulary, rendered in 2.5D by EXTRUDING the
// 2D SDF footprints into 3D (extrude helper below), plus a true 3D sphere.
//   kind 0 = sphere   (TRUE 3D, length(p) - r)
//   kind 1 = box           (extruded rectangle)
//   kind 2 = cylinder      (extruded circle)
//   kind 3 = rhombus       (extruded)
//   kind 4 = parallelogram (extruded)
//   kind 5 = trapezoid     (extruded)
//   kind 6 = triangle      (extruded)
//
// Camera is orthographic, so every camera ray is parallel: ray direction is a
// constant uniform (uRayDir), only the per-fragment origin varies.

export const shaderCode = /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  rayDir   : vec4<f32>,   // xyz = world-space camera forward (normalized); w unused
  lightDir : vec4<f32>,   // xyz = world-space light direction (points along travel); w unused
};

// Per-instance data (std140 storage, 5 x vec4 = 80 bytes).
struct Inst {
  centerYaw : vec4<f32>,  // xyz = world center, w = yaw (rotation about Z)
  halfKindR : vec4<f32>,  // x = hx (footprint half-width), y = hy, z = roundness, w = kind
  values01h : vec4<f32>,  // x = values[0], y = values[1], z = height, w = pad
  values234 : vec4<f32>,  // x = values[2], y = values[3], z = values[4], w = values[5]
  color     : vec4<f32>,  // rgb = albedo, a unused
};

@group(0) @binding(0) var<uniform> uni : Uniforms;
@group(0) @binding(1) var<storage, read> insts : array<Inst>;

struct VOut {
  @builtin(position) clip  : vec4<f32>,
  @location(0) @interpolate(flat) ii : u32,
  @location(1) world : vec3<f32>,
};

struct FOut {
  @location(0) color : vec4<f32>,
  @builtin(frag_depth) depth : f32,
};

fn rotZ(p : vec2<f32>, c : f32, s : f32) -> vec2<f32> {
  return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

// ---------------------------------------------------------------------------
// 2D SDF footprints (ported verbatim from the source 2D shader) + helpers.
// ---------------------------------------------------------------------------

fn ndot(a : vec2<f32>, b : vec2<f32>) -> f32 {
  return a.x * b.x - a.y * b.y;
}

fn dot2(v : vec2<f32>) -> f32 {
  return dot(v, v);
}

fn op_round(d : f32, r : f32) -> f32 {
  return d - r;
}

fn sd_circle(p : vec2<f32>, r : f32) -> f32 {
  return length(p) - r;
}

fn sd_rectangle(p : vec2<f32>, w : f32, h : f32) -> f32 {
  let b = vec2<f32>(w, h);
  let d = abs(p) - b;
  return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sd_rhombus(_p : vec2<f32>, wi : f32, he : f32) -> f32 {
  let p = abs(_p);
  let b = vec2<f32>(wi, he);
  let h = clamp(ndot(b - 2.0 * p, b) / dot(b, b), -1.0, 1.0);
  let d = length(p - 0.5 * b * vec2<f32>(1.0 - h, 1.0 + h));
  return d * sign(p.x * b.y + p.y * b.x - b.x * b.y);
}

fn sd_parallelogram(p : vec2<f32>, wi : f32, he : f32, sk : f32) -> f32 {
  let e = vec2<f32>(sk, he);
  let e2 = sk * sk + he * he;
  var pos = select(p, -p, p.y < 0.0);
  // Horizontal edge
  var w = pos - e;
  w.x = w.x - clamp(w.x, -wi, wi);
  var d = vec2<f32>(dot(w, w), -w.y);
  // Vertical edge
  let s = pos.x * e.y - pos.y * e.x;
  pos = select(pos, -pos, s < 0.0);
  var v = pos - vec2<f32>(wi, 0.0);
  v = v - e * clamp(dot(v, e) / e2, -1.0, 1.0);
  d = min(d, vec2<f32>(dot(v, v), wi * he - abs(s)));
  return sqrt(d.x) * sign(-d.y);
}

fn sd_trapezoid(p : vec2<f32>, r1 : f32, r2 : f32, he : f32) -> f32 {
  let k1 = vec2<f32>(r2, he);
  let k2 = vec2<f32>(r2 - r1, 2.0 * he);
  var pp = p;
  pp.x = abs(pp.x);
  let selected_r = select(r2, r1, p.y < 0.0);
  let ca = vec2<f32>(max(0.0, pp.x - selected_r), abs(pp.y) - he);
  let cb = pp - k1 + k2 * clamp(dot(k1 - pp, k2) / dot2(k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot2(ca), dot2(cb)));
}

fn sd_triangle(p : vec2<f32>, p0 : vec2<f32>, p1 : vec2<f32>, p2 : vec2<f32>) -> f32 {
  let e0 = p1 - p0;
  let e1 = p2 - p1;
  let e2 = p0 - p2;

  let v0 = p - p0;
  let v1 = p - p1;
  let v2 = p - p2;

  let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);

  let s = sign(e0.x * e2.y - e0.y * e2.x);

  let d0 = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
  let d1 = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
  let d2 = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));

  let d = min(min(d0, d1), d2);

  return -sqrt(d.x) * sign(d.y);
}

// Dispatch the 2D footprint SDF by kind. Each non-sphere kind reuses the EXACT
// 2D sd_* function above + op_round(roundness) exactly as the source does it:
// roundness is subtracted from the shape extents BEFORE the sd_* call, then
// op_round() is applied to the returned distance.
fn sd_2d_for_kind(p : vec2<f32>, hx : f32, hy : f32, roundness : f32, values : array<f32, 6>, kind : f32) -> f32 {
  let r = roundness;
  if (kind < 1.5) {
    // kind 1: box (rectangle) — values[0] = hx, values[1] = hy
    let d = sd_rectangle(p, values[0] - r, values[1] - r);
    return op_round(d, r);
  } else if (kind < 2.5) {
    // kind 2: cylinder (circle) — values[0] = radius
    let d = sd_circle(p, values[0] - r);
    return op_round(d, r);
  } else if (kind < 3.5) {
    // kind 3: rhombus — values[0] = hx, values[1] = hy
    let d = sd_rhombus(p, values[0] - r, values[1] - r);
    return op_round(d, r);
  } else if (kind < 4.5) {
    // kind 4: parallelogram — values[0] = hx, values[1] = hy, values[2] = skew
    let d = sd_parallelogram(p, values[0] - r, values[1] - r, values[2]);
    return op_round(d, r);
  } else if (kind < 5.5) {
    // kind 5: trapezoid — values[0] = r1, values[1] = r2, values[2] = he
    let d = sd_trapezoid(p, values[0] - r, values[1] - r, values[2] - r);
    return op_round(d, r);
  } else {
    // kind 6: triangle — derive three vertices from the hx/hy footprint, with
    // roundness pulled in per-vertex via sign() to preserve direction (as the
    // source does), then op_round() applied after.
    let ax = 0.0;
    let ay = values[1] - sign(values[1]) * r;
    let bx = -(values[0] - sign(values[0]) * r);
    let by = -(values[1] - sign(values[1]) * r);
    let cx = values[0] - sign(values[0]) * r;
    let cy = -(values[1] - sign(values[1]) * r);
    let d = sd_triangle(p, vec2<f32>(ax, ay), vec2<f32>(bx, by), vec2<f32>(cx, cy));
    return op_round(d, r);
  }
}

// ---------------------------------------------------------------------------
// 3D SDF: kind 0 = true sphere; kinds 1..6 = extruded 2D footprints.
// ---------------------------------------------------------------------------

fn extrude(d2 : f32, pz : f32, half_h : f32) -> f32 {
  let w = vec2<f32>(d2, abs(pz) - half_h);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2<f32>(0.0)));
}

fn sdf(p : vec3<f32>, hx : f32, hy : f32, hz : f32, roundness : f32, values : array<f32, 6>, kind : f32) -> f32 {
  if (kind < 0.5) {
    return length(p) - hx;   // kind 0: true 3D sphere (hx = radius)
  }
  let d2 = sd_2d_for_kind(p.xy, hx, hy, roundness, values, kind);
  return extrude(d2, p.z, hz);
}

fn sdfNormal(p : vec3<f32>, hx : f32, hy : f32, hz : f32, roundness : f32, values : array<f32, 6>, kind : f32) -> vec3<f32> {
  let e = vec2<f32>(0.0015, -0.0015);
  return normalize(
    e.xyy * sdf(p + e.xyy, hx, hy, hz, roundness, values, kind) +
    e.yyx * sdf(p + e.yyx, hx, hy, hz, roundness, values, kind) +
    e.yxy * sdf(p + e.yxy, hx, hy, hz, roundness, values, kind) +
    e.xxx * sdf(p + e.xxx, hx, hy, hz, roundness, values, kind)
  );
}

@vertex
fn vs(@location(0) corner : vec3<f32>, @builtin(instance_index) ii : u32) -> VOut {
  let inst = insts[ii];
  let center = inst.centerYaw.xyz;
  let yaw = inst.centerYaw.w;
  let hx = inst.halfKindR.x;
  let hy = inst.halfKindR.y;
  let hz = inst.values01h.z / 2.0;   // height / 2

  // Unit cube corner (-1..1) scaled by footprint half-extents + half-height,
  // then rotated into world. The XY half-extents are the footprint's bounding
  // box (see boundsLogic in main.ts), so the silhouette never clips.
  let scaled = corner * vec3<f32>(hx, hy, hz);
  let xy = rotZ(scaled.xy, cos(yaw), sin(yaw));
  let world = center + vec3<f32>(xy, scaled.z);

  var out : VOut;
  out.clip = uni.viewProj * vec4<f32>(world, 1.0);
  out.ii = ii;
  out.world = world;
  return out;
}

@fragment
fn fs(in : VOut) -> FOut {
  let inst = insts[in.ii];
  let center = inst.centerYaw.xyz;
  let yaw = inst.centerYaw.w;
  let hx = inst.halfKindR.x;
  let hy = inst.halfKindR.y;
  let hz = inst.values01h.z / 2.0;   // height / 2
  let roundness = inst.halfKindR.z;
  let kind = inst.halfKindR.w;
  let values = array<f32, 6>(
    inst.values01h.x, inst.values01h.y,
    inst.values234.x, inst.values234.y, inst.values234.z, inst.values234.w
  );

  // World ray (origin = this fragment's box surface point) → instance local space.
  // Local space removes the yaw so the SDF stays axis-aligned and distance-correct
  // (rigid transform only, no scale → distances are preserved).
  let ic = cos(-yaw);
  let is = sin(-yaw);
  let relW = in.world - center;
  let lo = vec3<f32>(rotZ(relW.xy, ic, is), relW.z);
  let ld = normalize(vec3<f32>(rotZ(uni.rayDir.xy, ic, is), uni.rayDir.z));

  // Slab test against the local AABB (footprint half-extents + half-height).
  // Robust regardless of which cube face produced this fragment.
  let half = vec3<f32>(hx, hy, hz);
  // Guard the reciprocal: lo sits exactly on an AABB face (in.world is a cube-surface
  // point), so for a near-axis-aligned ray (ld component ~0 as the camera orbits)
  // (half - lo) * (1/ld) would be 0 * Inf = NaN and corrupt the slab reduction.
  let safeLd = select(ld, vec3<f32>(1e-6), abs(ld) < vec3<f32>(1e-6));
  let inv = 1.0 / safeLd;
  let tA = (-half - lo) * inv;
  let tB = (half - lo) * inv;
  let tmin = min(tA, tB);
  let tmax = max(tA, tB);
  let t0 = max(max(tmin.x, tmin.y), tmin.z);
  let t1 = min(min(tmax.x, tmax.y), tmax.z);
  if (t1 < max(t0, 0.0)) {
    discard;
  }

  // Sphere-trace between entry and exit.
  var t = max(t0, 0.0);
  var hit = false;
  for (var i = 0; i < 96; i = i + 1) {
    let d = sdf(lo + ld * t, hx, hy, hz, roundness, values, kind);
    if (d < 0.001) {
      hit = true;
      break;
    }
    t = t + d;
    if (t > t1) {
      break;
    }
  }
  if (!hit) {
    discard;
  }

  let pLocal = lo + ld * t;
  let nLocal = sdfNormal(pLocal, hx, hy, hz, roundness, values, kind);

  // Back to world (forward yaw).
  let fc = cos(yaw);
  let fs_ = sin(yaw);
  let nWorld = normalize(vec3<f32>(rotZ(nLocal.xy, fc, fs_), nLocal.z));
  let pWorld = center + vec3<f32>(rotZ(pLocal.xy, fc, fs_), pLocal.z);

  let clip = uni.viewProj * vec4<f32>(pWorld, 1.0);

  // Plausible shading: ambient + clamped diffuse against one directional light.
  let diff = max(dot(nWorld, -uni.lightDir.xyz), 0.0);
  let ambient = 0.28;
  let shade = ambient + (1.0 - ambient) * diff;

  var out : FOut;
  out.color = vec4<f32>(inst.color.rgb * shade, 1.0);
  out.depth = clip.z / clip.w;
  return out;
}
`;
