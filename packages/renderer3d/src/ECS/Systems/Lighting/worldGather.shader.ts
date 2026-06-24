import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { MAX_INSTANCE_COUNT } from "../SDFSystem/sdf.shader.ts";
import { sceneSDF } from "../SDFSystem/sceneSDF.wgsl.ts";

// Stage-1 world-space Radiance Cascades — GATHER pass (doc §6, §3.4).
//
// ONE cascade (c0). Fullscreen draw over the `probeRad` atlas texture: every
// fragment is one (probe, direction) cell. We decode the probe index (i,j) and
// the octahedral direction cell (u,v) from the integer fragment coordinate,
// place the probe at its camera-snapped world XY on the probe plane, oct_decode
// the ray direction (full sphere), then sphere-trace the world SDF scene along
// the SINGLE cascade interval [intervalStart, intervalEnd] (world units).
//
// scene_sdf = min over instances of each instance's LOCAL sd_shape3d (world->local
// = subtract center, inverse yaw), reusing the SHARED helpers from sceneSDF.wgsl.ts
// and the SAME per-instance storage buffers createDrawShapeSystem fills each frame
// (uTransform/uKind/uValues/uRoundness/uHeights/uColor/uMaterial).
//
// Output (rgba16float atlas):
//   rgb = interval radiance — emission of the hit instance (material.x>0), else
//         sky/sun on a miss (single cascade => the top cascade => always blends sky).
//   a   = visibility — 1.0 if the ray passed the whole interval unobstructed, else 0.
//
// COORDINATES: world is Z-up, footprints in XY, probe plane at uProbePlaneZ. The
// gather works purely in world space (no viewProj) — it needs only the grid origin.
// The atlas is fixed-size (zoom-independent BY CONSTRUCTION); GRID_DIM/DIR0_W are
// baked as overridable WGSL constants so the (probe,dir) decode matches the texture.

// Atlas layout constants (must agree with createRCTextures' WORLD_GRID_DIM/WORLD_DIR0_W
// and with createWorldRadianceCascadesSystem). Probe (i,j) owns the DIR0_W x DIR0_W
// tile at pixel origin (i*DIR0_W, j*DIR0_W); within it (u,v) is the octahedral cell.
export const WORLD_GRID_DIM = 128;
export const WORLD_DIR0_W = 4;

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : per-frame uniforms ----
    // Camera-snapped grid origin (world XY); .zw unused.
    gridOrigin: new VariableMeta("uGridOrigin", VariableKind.Uniform, `vec4<f32>`),
    // World height of the (single, ground) probe plane.
    probePlaneZ: new VariableMeta("uProbePlaneZ", VariableKind.Uniform, `f32`),
    // This cascade's probes-per-side and octahedral tile side (gridDim*dirW = atlas
    // side, constant across cascades). Decode + probe placement read these so one
    // shader serves any cascade.
    gridDim: new VariableMeta("uGridDim", VariableKind.Uniform, `u32`),
    dirW: new VariableMeta("uDirW", VariableKind.Uniform, `u32`),
    // World units per probe at THIS cascade (cell0 * 2^c).
    cell: new VariableMeta("uCell", VariableKind.Uniform, `f32`),
    // This cascade's interval [start, end] in world units.
    intervalStart: new VariableMeta("uIntervalStart", VariableKind.Uniform, `f32`),
    intervalEnd: new VariableMeta("uIntervalEnd", VariableKind.Uniform, `f32`),
    // Sphere-trace step budget (~32..48).
    gatherSteps: new VariableMeta("uGatherSteps", VariableKind.Uniform, `f32`),
    // Live scene instance count (<= MAX_INSTANCE_COUNT).
    instanceCount: new VariableMeta("uInstanceCount", VariableKind.Uniform, `u32`),
    // Sun direction toward the sun (radians, screen frame +X right +Y down) and
    // enabled flag, mirrored from SunLight each frame; .y = enabled (0/1).
    sun: new VariableMeta("uSun", VariableKind.Uniform, `vec4<f32>`),
    // Directional light color, .w = softness (matches radianceCascades uSunColor).
    sunColor: new VariableMeta("uSunColor", VariableKind.Uniform, `vec4<f32>`),
    // Sky fill color, .w = skyMix (matches radianceCascades uSkyColor).
    skyColor: new VariableMeta("uSkyColor", VariableKind.Uniform, `vec4<f32>`),

    // ---- group 1 : per-instance storage (SAME types/names as sdf.shader.ts) ----
    // Bound to the SAME GPUBuffers the draw system fills — declare identical layout
    // so the shared sceneSDF helpers (which read uKind/uValues/uRoundness by global
    // name) and the scene_sdf loop below see the live scene.
    transform: new VariableMeta(
      "uTransform",
      VariableKind.StorageRead,
      `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
    kind: new VariableMeta("uKind", VariableKind.StorageRead, `array<u32, ${MAX_INSTANCE_COUNT}>`),
    values: new VariableMeta(
      "uValues",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT * 6}>`,
    ),
    roundness: new VariableMeta(
      "uRoundness",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
    heights: new VariableMeta(
      "uHeights",
      VariableKind.StorageRead,
      `array<f32, ${MAX_INSTANCE_COUNT}>`,
    ),
    color: new VariableMeta(
      "uColor",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
    // Emission material: .x = intensity (>0 emitter), packed like sdf.shader.ts.
    material: new VariableMeta(
      "uMaterial",
      VariableKind.StorageRead,
      `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`,
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const POSITION = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );

struct VertexOutput {
  @builtin(position) position: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(POSITION[vertexIndex], 0.0, 1.0);
  return output;
}

const PI: f32 = 3.14159265;
const TAU: f32 = 2.0 * PI;

// Shared local-SDF helpers (rotZ, sd_*, sd_2d_for_kind, extrude, sd_shape3d,
// footprint_half_xy). They read uKind/uValues/uRoundness by global name, declared
// in group 1 above with identical types.
${sceneSDF}

// ===== Octahedral (doc §3.4) =====

fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
  let e = e_in;
  var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) {
    v = vec3<f32>((1.0 - abs(v.yx)) * sign(v.xy), v.z);
  }
  return normalize(v);
}

// Probe (i,j) world XY for the single cascade (c = 0): camera-snapped origin plus
// the cell-centered offset from the grid center (doc §3.2).
fn probe_world_xy(ij: vec2<u32>) -> vec2<f32> {
  return uGridOrigin.xy
       + (vec2<f32>(ij) + 0.5 - f32(uGridDim) * 0.5) * uCell;
}

// Directional source + faint sky fill, parameterized by the ray's screen-frame
// angle (same blend as radianceCascades.shader.ts sunAndSky). On a single cascade
// this top-cascade term is added on every miss.
fn sun_and_sky(rayAngle: f32) -> vec3f {
  let sky = uSkyColor.rgb * uSkyColor.w;
  if (uSun.y < 0.5) {
    // Sun disabled: sky fill only.
    return sky;
  }
  let angleToSun = rayAngle - uSun.x;
  let c = cos(angleToSun);
  let sunIntensity = pow(max(0.0, c), 4.0 / max(1e-4, uSunColor.w));
  return mix(uSunColor.rgb * sunIntensity, uSkyColor.rgb, uSkyColor.w);
}

// scene_sdf = nearest instance's local SDF (world->local: subtract center, inverse
// yaw, then sd_shape3d). Tracks the hit instance for emission lookup (doc §6.1).
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

// Emission of a hit instance. material.x encodes intensity; its SIGN is a flag
// (negative = directional emitter, see sdf.shader.ts fs_emit), so brightness is
// abs(intensity). intensity == 0 is a pure occluder (no radiance, only blocks
// visibility). Directional cone shaping is out of scope for Stage 1 — the beam
// just must not vanish.
fn emission_of(instance: u32) -> vec3<f32> {
  let intensity = uMaterial[instance].x;
  if (intensity != 0.0) {
    return uColor[instance].rgb * abs(intensity);
  }
  return vec3<f32>(0.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
  // 1. Decode the (probe, direction) cell from the integer atlas pixel.
  let px = vec2<u32>(floor(frag_coord.xy));
  let ij = px / uDirW;          // probe index
  let uv = px % uDirW;          // direction cell within the tile

  let ro0 = vec3<f32>(probe_world_xy(ij), uProbePlaneZ);
  let oct = (( vec2<f32>(uv) + 0.5) / f32(uDirW)) * 2.0 - 1.0;
  let dir = oct_decode(oct);

  // 2. World-space interval along the ray (single cascade c0).
  let ro = ro0 + dir * uIntervalStart;
  let tmax = uIntervalEnd - uIntervalStart;

  // 3. Sphere-trace scene_sdf, tracking the nearest hit instance.
  var t = 0.0;
  var hit = false;
  var hitInstance: u32 = 0u;
  let steps = i32(uGatherSteps);
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

  // 4. Interval radiance + visibility.
  var radiance = vec3<f32>(0.0);
  let visibility = select(0.0, 1.0, !hit);
  if (hit) {
    radiance = emission_of(hitInstance);
  } else {
    // Single cascade => this IS the top cascade: blend sky/sun on every miss.
    // Screen-frame ray angle from the direction's XY (matches sunAndSky's frame:
    // +X right, +Y down => negate dir.y, same as the cascade raymarch's rayDir).
    let rayAngle = atan2(-dir.y, dir.x);
    radiance = sun_and_sky(rayAngle);
  }

  return vec4f(radiance, visibility);
}
    `,
);
