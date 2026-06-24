// WGSL for the 2.5D SDF impostor prototype.
//
// One draw call, instanced. Per instance we rasterize a world-space bounding
// box (the "impostor") and, in the fragment shader, sphere-trace the instance's
// local 3D SDF (sphere or box) along the camera ray. The ray hit gives a real
// world-space point → we write its clip depth to frag_depth, so the depth buffer
// sorts/occludes everything correctly with zero CPU sorting. A single directional
// light shades the surface normal so volumes read as plausibly 3D.
//
// Camera is orthographic, so every camera ray is parallel: ray direction is a
// constant uniform (uRayDir), only the per-fragment origin varies.

export const shaderCode = /* wgsl */ `
struct Uniforms {
  viewProj : mat4x4<f32>,
  rayDir   : vec4<f32>,   // xyz = world-space camera forward (normalized); w unused
  lightDir : vec4<f32>,   // xyz = world-space light direction (points along travel); w unused
};

// kind: 0 = sphere, 1 = box. half = local half-extents (sphere uses half.x as radius).
struct Inst {
  centerYaw : vec4<f32>,  // xyz = world center, w = yaw (rotation about Z)
  halfKind  : vec4<f32>,  // xyz = half extents, w = kind
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

@vertex
fn vs(@location(0) corner : vec3<f32>, @builtin(instance_index) ii : u32) -> VOut {
  let inst = insts[ii];
  let center = inst.centerYaw.xyz;
  let yaw = inst.centerYaw.w;
  let half = inst.halfKind.xyz;

  // Unit cube corner (-1..1) scaled by half-extents and rotated into world.
  let scaled = corner * half;
  let xy = rotZ(scaled.xy, cos(yaw), sin(yaw));
  let world = center + vec3<f32>(xy, scaled.z);

  var out : VOut;
  out.clip = uni.viewProj * vec4<f32>(world, 1.0);
  out.ii = ii;
  out.world = world;
  return out;
}

fn sdf(p : vec3<f32>, half : vec3<f32>, kind : f32) -> f32 {
  if (kind < 0.5) {
    return length(p) - half.x;            // sphere
  }
  let q = abs(p) - half;                   // box
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdfNormal(p : vec3<f32>, half : vec3<f32>, kind : f32) -> vec3<f32> {
  let e = vec2<f32>(0.0015, -0.0015);
  return normalize(
    e.xyy * sdf(p + e.xyy, half, kind) +
    e.yyx * sdf(p + e.yyx, half, kind) +
    e.yxy * sdf(p + e.yxy, half, kind) +
    e.xxx * sdf(p + e.xxx, half, kind)
  );
}

@fragment
fn fs(in : VOut) -> FOut {
  let inst = insts[in.ii];
  let center = inst.centerYaw.xyz;
  let yaw = inst.centerYaw.w;
  let half = inst.halfKind.xyz;
  let kind = inst.halfKind.w;

  // World ray (origin = this fragment's box surface point) → instance local space.
  // Local space removes the yaw so the SDF stays axis-aligned and distance-correct
  // (rigid transform only, no scale → distances are preserved).
  let ic = cos(-yaw);
  let is = sin(-yaw);
  let relW = in.world - center;
  let lo = vec3<f32>(rotZ(relW.xy, ic, is), relW.z);
  let ld = normalize(vec3<f32>(rotZ(uni.rayDir.xy, ic, is), uni.rayDir.z));

  // Slab test against the local AABB (half-extents). Robust regardless of which
  // cube face produced this fragment.
  let inv = 1.0 / ld;
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
    let d = sdf(lo + ld * t, half, kind);
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
  let nLocal = sdfNormal(pLocal, half, kind);

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
