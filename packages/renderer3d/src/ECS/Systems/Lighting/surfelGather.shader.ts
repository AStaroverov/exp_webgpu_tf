import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { MAX_INSTANCE_COUNT } from "../SDFSystem/sdf.shader.ts";
import { sceneSDF } from "../SDFSystem/sceneSDF.wgsl.ts";
import { SURFEL_CAP, SURFEL_DIR0_W, SURFEL_DIR_COUNT } from "./surfelResources.ts";

// Surfel Radiance Cascades — Stage C GATHER compute pass (doc §7).
//
// One thread per surfel slot (@workgroup_size(64), dispatch ceil(CAP/64)). Each
// LIVE surfel (posr.w > 0) sphere-traces the world SDF scene in SURFEL_DIR_COUNT
// octahedral directions (full sphere) from its position lifted along its normal,
// and stores per-direction radiance + visibility into surfel_rad. This is the
// SAME trace as worldGather, but the ray ORIGIN is the surfel position (one tile
// per surfel) instead of a probe-grid point.
//
// scene_sdf = min over instances of each instance's LOCAL sd_shape3d (world->local
// = subtract center, inverse yaw), reusing the SHARED helpers from sceneSDF.wgsl.ts
// and the SAME per-instance storage buffers createDrawShapeSystem fills each frame
// (uTransform/uKind/uValues/uRoundness/uHeights/uColor/uMaterial).
//
// surfel_rad output (per direction cell, index id*DIR_COUNT + v*DIR0_W + u):
//   rgb = interval radiance — emission of the hit instance (material.x != 0), else
//         sky/sun on a miss (single cascade => top cascade => always blends sky).
//   a   = visibility — 1.0 if the ray passed unobstructed (miss), else 0.0.
//
// COORDINATES: world is Z-up, footprints in XY. sun_and_sky uses a SCREEN-frame ray
// angle (matches worldGather: +X right, +Y down => negate dir.y).
//
// surfel_posr / surfel_norw / surfel_rad are STANDALONE GPUBuffers (see
// surfelResources.ts); the VariableMetas below exist ONLY for WGSL emission + the
// (kind-based, size-agnostic) bind-group layout. Bind them manually against
// pipeline.getBindGroupLayout(g) (autoLayout). The 7 instance buffers are the draw
// system's GPUVariables — bind via their .getBindGroupEntry(device).

// COMPUTE-visibility group-0 uniform.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, {
    visibility: GPUShaderStage.COMPUTE,
  });

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms (COMPUTE-only) ----
    // Packed gather scalars (one buffer to stay under the 12-uniform-per-stage
    // limit): .x = interval start, .y = gather distance, .z = sphere-trace step
    // budget, .w = normal bias (ray-origin lift along the surfel normal).
    params: uC("uParams", `vec4<f32>`),
    // Live scene instance count (<= MAX_INSTANCE_COUNT).
    instanceCount: uC("uInstanceCount", `u32`),
    // Sun direction toward the sun (radians, screen frame +X right +Y down) and
    // enabled flag, mirrored from SunLight each frame; .y = enabled (0/1).
    sun: uC("uSun", `vec4<f32>`),
    // Directional light color, .w = softness (matches worldGather uSunColor).
    sunColor: uC("uSunColor", `vec4<f32>`),
    // Sky fill color, .w = skyMix (matches worldGather uSkyColor).
    skyColor: uC("uSkyColor", `vec4<f32>`),

    // ---- group 1 : storage reads (StorageRead => @group(1)) ----
    // Surfel-read bufs FIRST (binding 0,1), then the 7 instance bufs (binding 2..8)
    // — binding order follows declaration order (setupVariable). The manual bind
    // group must match: surfel_posr, surfel_norw, then instance bufs in this order.
    //
    // surfel_posr: xyz = world position, w = radius² (w == 0 => DEAD slot).
    surfelPosr: new VariableMeta(
      "surfel_posr",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    // surfel_norw: xyz = surface normal, w = recycle marker (unused here).
    surfelNorw: new VariableMeta(
      "surfel_norw",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${SURFEL_CAP}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),

    // Per-instance scene storage — SAME names/types as sdf.shader.ts / worldGather,
    // so the shared sceneSDF helpers (which read uKind/uValues/uRoundness by global
    // name) and the scene_sdf loop below see the live scene. Bound to the draw
    // system's GPUVariables via .getBindGroupEntry(device) in THIS order.
    transform: new VariableMeta(
      "uTransform",
      VariableKind.StorageRead,
      `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    kind: new VariableMeta("uKind", VariableKind.StorageRead, `array<u32, ${MAX_INSTANCE_COUNT}>`, {
      visibility: GPUShaderStage.COMPUTE,
    }),
    values: new VariableMeta(
      "uValues",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT * 6}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    roundness: new VariableMeta(
      "uRoundness",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    heights: new VariableMeta(
      "uHeights",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    color: new VariableMeta(
      "uColor",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
    // Emission material: .x = intensity (!= 0 => emitter), packed like sdf.shader.ts.
    material: new VariableMeta(
      "uMaterial",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),

    // ---- group 2 : surfel radiance cache write (StorageWrite => @group(2)) ----
    // vec4<f32>[CAP * DIR_COUNT]. rgb = direction radiance, a = visibility.
    // Index: surfel_rad[id * DIR_COUNT + (v*DIR0_W + u)].
    surfelRad: new VariableMeta(
      "surfel_rad",
      VariableKind.StorageWrite,
      `array<vec4<f32>, ${SURFEL_CAP * SURFEL_DIR_COUNT}>`,
      { visibility: GPUShaderStage.COMPUTE },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const PI: f32 = 3.14159265;
const DIR0_W: u32 = ${SURFEL_DIR0_W}u;
const DIR_COUNT: u32 = ${SURFEL_DIR_COUNT}u;

// Shared local-SDF helpers (rotZ, sd_*, sd_2d_for_kind, extrude, sd_shape3d,
// footprint_half_xy). They read uKind/uValues/uRoundness by global name, declared
// in group 1 above with identical types.
${sceneSDF}

// ===== Octahedral (doc §3.4) — verbatim from worldGather =====
fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
  let e = e_in;
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    v = vec3<f32>((1.0 - abs(v.yx)) * sign(v.xy), v.z);
  }
  return normalize(v);
}

// Solid integer hash for cheap per-thread jitter seeds (not used for grid here;
// provided for parity with worldGather-style seeding if needed).
fn wang_hash(s: u32) -> u32 {
  var x = s;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x * 9u;
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

// Directional source + faint sky fill, parameterized by the ray's screen-frame
// angle (verbatim from worldGather sun_and_sky). On a single cascade this
// top-cascade term is added on every miss.
fn sun_and_sky(rayAngle: f32) -> vec3f {
  let sky = uSkyColor.rgb * uSkyColor.w;
  if (uSun.y < 0.5) {
    return sky;
  }
  let angleToSun = rayAngle - uSun.x;
  let c = cos(angleToSun);
  let sunIntensity = pow(max(0.0, c), 4.0 / max(1e-4, uSunColor.w));
  return mix(uSunColor.rgb * sunIntensity, uSkyColor.rgb, uSkyColor.w);
}

// scene_sdf = nearest instance's local SDF (world->local: subtract center, inverse
// yaw, then sd_shape3d). Tracks the hit instance for emission lookup. Verbatim
// from worldGather.
struct Hit { dist: f32, instance: u32 };

fn scene_sdf(p: vec3<f32>) -> Hit {
  var best = Hit(1e30, 0u);
  let n = min(uInstanceCount, ${MAX_INSTANCE_COUNT}u);
  for (var k: u32 = 0u; k < n; k = k + 1u) {
    let tr = uTransform[k];
    let hz = uHeights[k] * 0.5;
    let center = vec3<f32>(tr[3].x, tr[3].y, tr[3].z + hz);
    let yaw = atan2(tr[0].y, tr[0].x);
    let rel = p - center;
    let lp = vec3<f32>(rotZ(rel.xy, cos(-yaw), sin(-yaw)), rel.z);
    let d = sd_shape3d(lp, k, hz);
    if (d < best.dist) {
      best = Hit(d, k);
    }
  }
  return best;
}

// Emission of a hit instance. material.x encodes intensity; SIGN is a flag, so
// brightness is abs(intensity). intensity == 0 is a pure occluder. Verbatim from
// worldGather.
fn emission_of(instance: u32) -> vec3<f32> {
  let intensity = uMaterial[instance].x;
  if (intensity != 0.0) {
    return uColor[instance].rgb * abs(intensity);
  }
  return vec3<f32>(0.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  // 1. Bounds check (dispatch is ceil-rounded; drop the over-edge threads).
  if (id >= ${SURFEL_CAP}u) {
    return;
  }

  // 2. Skip dead slots (posr.w == 0). Live surfels have w = radius² > 0.
  let posr = surfel_posr[id];
  if (posr.w <= 0.0) {
    return;
  }

  let pos = posr.xyz;
  let N = normalize(surfel_norw[id].xyz);

  // Unpack gather params.
  let intervalStart = uParams.x;
  let gatherDistance = uParams.y;
  let steps = i32(uParams.z);
  let normalBias = uParams.w;

  // 3. Lift the ray origin off the surface so rays don't self-hit at t == 0.
  let ro0 = pos + N * normalBias;

  // 4. One sphere-trace per octahedral direction (full sphere).
  let tmax = gatherDistance - intervalStart;
  for (var v: u32 = 0u; v < DIR0_W; v = v + 1u) {
    for (var u: u32 = 0u; u < DIR0_W; u = u + 1u) {
      let oct = ((vec2<f32>(f32(u), f32(v)) + 0.5) / f32(DIR0_W)) * 2.0 - 1.0;
      let dir = oct_decode(oct);
      let ro = ro0 + dir * intervalStart;

      var t = 0.0;
      var hit = false;
      var hitInstance: u32 = 0u;
      for (var s = 0; s < steps; s = s + 1) {
        let h = scene_sdf(ro + dir * t);
        if (h.dist < 0.001) {
          hit = true;
          hitInstance = h.instance;
          break;
        }
        t = t + h.dist;
        if (t > tmax) {
          break;
        }
      }

      var radiance = vec3<f32>(0.0);
      let visibility = select(0.0, 1.0, !hit);
      if (hit) {
        radiance = emission_of(hitInstance);
      } else {
        // Single cascade => top cascade: blend sky/sun on every miss. Screen-frame
        // ray angle (matches worldGather: +X right, +Y down => negate dir.y).
        let rayAngle = atan2(-dir.y, dir.x);
        radiance = sun_and_sky(rayAngle);
      }

      surfel_rad[id * DIR_COUNT + (v * DIR0_W + u)] = vec4<f32>(radiance, visibility);
    }
  }
}
    `,
);
