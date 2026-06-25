import { VariableKind, VariableMeta } from "../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../WGSL/wgsl.ts";
import { MAX_INSTANCE_COUNT } from "../SDFSystem/sdf.shader.ts";
import { sceneSDF } from "../SDFSystem/sceneSDF.wgsl.ts";

// Voxelization compute pass — one thread per voxel. The voxel center is mapped to world
// space, the scene SDF (min over all instances, reusing the SAME shared helpers + scene
// buffers as the draw/gather passes) is evaluated, and a voxel that the surface crosses
// is filled with the nearest instance's albedo + emission. Output goes to two 3D storage
// textures (group 2).
//
// COORDINATES: world is Z-up, footprints in XY (identical to sceneSDF / the gather pass).
// A voxel is "solid" when |sdf| at its center <= half its diagonal (cellSize*0.5*sqrt(3));
// that's the conservative test for "the iso-surface passes through this cell".

// COMPUTE-visibility group-0 uniform helper.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

// StorageRead scene buffer (group 1), COMPUTE-visible. Bound to the draw system's
// GPUVariables (sceneInstances) — no data copy.
const sceneBuf = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.StorageRead, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 4; // 4*4*4 = 64 threads/workgroup

export const shaderMeta = new ShaderMeta(
  {
    // ---- group 0 : grid uniforms (COMPUTE-only) ----
    // .xyz = world min corner of the grid box, .w = cellSize (world units per voxel).
    gridOrigin: uC("uGridOrigin", `vec4<f32>`),
    // .xyz = voxel counts per axis (i32 for direct bounds compare), .w unused.
    gridDims: uC("uGridDims", `vec4<i32>`),
    // Live scene instance count (<= MAX_INSTANCE_COUNT).
    instanceCount: uC("uInstanceCount", `u32`),
    // Directional sun: .xyz = normalized world dir TOWARD sun, .w = effective intensity
    // (0 when the sun is disabled).
    sun: uC("uSun", `vec4<f32>`),
    // .rgb = sun color (linear).
    sunColor: uC("uSunColor", `vec4<f32>`),

    // ---- group 1 : per-instance scene storage (StorageRead => @group(1)) ----
    // SAME names/types as sdf.shader.ts / the former gather, so the shared sceneSDF
    // helpers (which read uKind/uValues/uRoundness by global name) see the live scene.
    // Bound to sceneInstances.* GPUVariables in THIS declaration order.
    transform: sceneBuf("uTransform", `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`),
    kind: sceneBuf("uKind", `array<u32, ${MAX_INSTANCE_COUNT}>`),
    values: sceneBuf("uValues", `array<f32, ${MAX_INSTANCE_COUNT * 6}>`),
    roundness: sceneBuf("uRoundness", `array<f32, ${MAX_INSTANCE_COUNT}>`),
    heights: sceneBuf("uHeights", `array<f32, ${MAX_INSTANCE_COUNT}>`),
    color: sceneBuf("uColor", `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`),
    material: sceneBuf("uMaterial", `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`),

    // ---- group 2 : voxel output (StorageTexture, write-only) ----
    voxelAlbedo: new VariableMeta(
      "voxelAlbedo",
      VariableKind.StorageTexture,
      `texture_storage_3d<rgba8unorm, write>`,
      {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba8unorm",
        storageTextureAccess: "write-only",
      },
    ),
    voxelEmission: new VariableMeta(
      "voxelEmission",
      VariableKind.StorageTexture,
      `texture_storage_3d<rgba16float, write>`,
      {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      },
    ),
    voxelRadiance: new VariableMeta(
      "voxelRadiance",
      VariableKind.StorageTexture,
      `texture_storage_3d<rgba16float, write>`,
      {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      },
    ),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const SQRT3: f32 = 1.7320508;

// Shared local-SDF helpers (rotZ, sd_*, sd_2d_for_kind, extrude, sd_shape3d, …). They
// read uKind/uValues/uRoundness by global name, declared in group 1 with identical types.
${sceneSDF}

// scene_sdf = nearest instance's local SDF (world->local: subtract center, inverse yaw,
// then sd_shape3d). Tracks the hit instance for albedo/emission lookup. Same as gather.
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

// Emission of an instance. material.x encodes intensity; SIGN is a flag, so brightness
// is abs(intensity). intensity == 0 is a pure occluder. Same as gather.
fn emission_of(instance: u32) -> vec3<f32> {
  let intensity = uMaterial[instance].x;
  if (intensity != 0.0) {
    return uColor[instance].rgb * abs(intensity);
  }
  return vec3<f32>(0.0);
}

// Surface normal at p via central differences of the scene SDF distance. eps = half a
// cell. Falls back to +Z when the gradient is degenerate (zero-length).
fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
  let eps = uGridOrigin.w * 0.5;
  let dx = vec3<f32>(eps, 0.0, 0.0);
  let dy = vec3<f32>(0.0, eps, 0.0);
  let dz = vec3<f32>(0.0, 0.0, eps);
  let g = vec3<f32>(
    scene_sdf(p + dx).dist - scene_sdf(p - dx).dist,
    scene_sdf(p + dy).dist - scene_sdf(p - dy).dist,
    scene_sdf(p + dz).dist - scene_sdf(p - dz).dist,
  );
  let len = length(g);
  if (len < 1e-6) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return g / len;
}

// Sun visibility (1 = lit, 0 = shadowed) at surface point p with normal N. We sphere-
// trace the ANALYTIC scene_sdf (NOT a voxel-DDA over voxelAlbedo) because the voxel
// storage textures are being WRITTEN this pass and cannot be sampled here.
fn sun_visibility(p: vec3<f32>, N: vec3<f32>) -> f32 {
  if (uSun.w <= 0.0) {
    return 1.0;
  }
  let cellSize = uGridOrigin.w;
  let minStep = cellSize * 0.5;
  let L = uSun.xyz;
  let maxDist = length(vec3<f32>(uGridDims.xyz) * cellSize);
  let ro = p + N * cellSize * 1.5;
  var t = 0.0;
  for (var i = 0; i < 128; i = i + 1) {
    let d = scene_sdf(ro + L * t).dist;
    if (d < minStep) {
      return 0.0;
    }
    t = t + max(d, minStep);
    if (t > maxDist) {
      break;
    }
  }
  return 1.0;
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec3<i32>(gid);
  // Dispatch is ceil-rounded; drop threads past the grid bounds.
  if (coord.x >= uGridDims.x || coord.y >= uGridDims.y || coord.z >= uGridDims.z) {
    return;
  }

  let cellSize = uGridOrigin.w;
  // Voxel center in world space.
  let world = uGridOrigin.xyz + (vec3<f32>(gid) + vec3<f32>(0.5)) * cellSize;

  let hit = scene_sdf(world);
  // voxelAlbedo / voxelEmission occupancy stays BINARY: the DDA debug/gi raymarchers test
  // a > 0.5, and the composite reads voxelEmission.rgb directly (full self-emission).
  // Conservative "iso-surface crosses this voxel": within half the voxel diagonal.
  let solid = hit.dist <= cellSize * 0.5 * SQRT3;

  // voxelRADIANCE (the only volume the cone samples, via its mip pyramid) uses ANTI-ALIASED
  // coverage instead: a linear band across the iso-surface (1 deep inside → 0.5 at the
  // centre → 0 at the outer extent, matching solid). Stored PREMULTIPLIED (rgb·coverage,
  // a=coverage) so (1) the premultiplied mip pyramid + cone over-operator stay consistent
  // and (2) a MOVING object's edge voxels fade smoothly instead of popping 0↔1 each frame —
  // this is what kills the voxel-grid shimmer/flicker when light passes near an object.
  let coverage = clamp(0.5 - hit.dist / (cellSize * SQRT3), 0.0, 1.0);

  if (solid) {
    let albedo = uColor[hit.instance].rgb;
    let N = sdf_normal(world);
    let L = uSun.xyz;
    let ndl = max(dot(N, L), 0.0);
    let vis = sun_visibility(world, N);
    let direct = albedo * (ndl * uSun.w * vis) * uSunColor.rgb;
    let radiance = direct + emission_of(hit.instance);
    textureStore(voxelAlbedo, coord, vec4<f32>(albedo, 1.0));
    textureStore(voxelEmission, coord, vec4<f32>(emission_of(hit.instance), 1.0));
    textureStore(voxelRadiance, coord, vec4<f32>(radiance * coverage, coverage));
  } else {
    textureStore(voxelAlbedo, coord, vec4<f32>(0.0));
    textureStore(voxelEmission, coord, vec4<f32>(0.0));
    textureStore(voxelRadiance, coord, vec4<f32>(0.0));
  }
}
`,
);
