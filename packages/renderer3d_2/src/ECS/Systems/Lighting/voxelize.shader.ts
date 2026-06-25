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
  // Conservative "iso-surface crosses this voxel": within half the voxel diagonal.
  let solid = hit.dist <= cellSize * 0.5 * SQRT3;

  if (solid) {
    let albedo = uColor[hit.instance].rgb;
    textureStore(voxelAlbedo, coord, vec4<f32>(albedo, 1.0));
    textureStore(voxelEmission, coord, vec4<f32>(emission_of(hit.instance), 1.0));
  } else {
    textureStore(voxelAlbedo, coord, vec4<f32>(0.0));
    textureStore(voxelEmission, coord, vec4<f32>(0.0));
  }
}
`,
);
