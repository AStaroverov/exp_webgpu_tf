import { VariableKind, VariableMeta } from "../../../../../Struct/VariableMeta.ts";
import { ShaderMeta } from "../../../../../WGSL/ShaderMeta.ts";
import { wgsl } from "../../../../../WGSL/wgsl.ts";
import { MAX_INSTANCE_COUNT } from "../../../SDFSystem/sdf.shader.ts";
import { sceneSDF } from "../../../SDFSystem/sceneSDF.wgsl.ts";

// Voxelization compute pass — SCATTER. Instead of one thread per voxel evaluating the
// WHOLE scene SDF (cost = NumVoxels x NumInstances, plus 6 more evals for the normal),
// each shape fills ONLY the voxels inside its own world AABB, evaluating ONLY its own
// local SDF. cost = SUM over instances of (voxels in its AABB) — asymptotically
// independent of the grid size and of the OTHER instances.
//
// The work is a flat 1D list: instance k owns the contiguous range
// [start[k], start[k] + n[k]) where n[k] = nx*ny*nz voxels of its AABB box. The CPU
// builds the per-instance AABB voxel boxes + the prefix-sum `start`; one 1D dispatch
// (2D-flattened to dodge the 65535 per-dim workgroup cap) walks the list, a per-thread
// binary search maps the global work index back to its owning instance, and the local
// offset decodes to a voxel coord inside that instance's box.
//
// COORDINATES: world is Z-up, footprints in XY (identical to sceneSDF / the gather pass).
// A voxel is "solid" when |sdf| at its center <= half its diagonal (cellSize*0.5*sqrt(3));
// that's the conservative test for "the iso-surface passes through this cell".
//
// Two @compute entry points share this module + the sceneSDF helpers:
//   clear : one thread per voxel over the FULL grid — zeroes the bound target volume.
//           The scatter pass only ever WRITES solid voxels (last-writer-wins on overlap),
//           so the volume MUST be cleared first.
//   main  : the scatter described above. Dispatched TWICE per frame with different targets:
//           emitters (uPass=1) into voxelEmission, then occluders (uPass=0) into
//           voxelRadiance mip 0, MERGING voxelEmission in (see the CLASS SPLIT note at main()).
//
// (A second clipmap level lived here briefly and was REMOVED: with the zoom-ladder cellSize the
// perceived far-field artifacts are scale-invariant and the coarse cascade had no visible effect
// in validation — not worth its volume pair + the two-level work list.)

// COMPUTE-visibility group-0 uniform helper.
const uC = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.Uniform, type, { visibility: GPUShaderStage.COMPUTE });

// StorageRead scene buffer (group 1), COMPUTE-visible. Bound to the draw system's
// GPUVariables (sceneInstances) — no data copy.
const sceneBuf = (name: string, type: string) =>
  new VariableMeta(name, VariableKind.StorageRead, type, { visibility: GPUShaderStage.COMPUTE });

export const WORKGROUP = 4; // clear pass: 4*4*4 = 64 threads/workgroup over the 3D grid
export const WORKGROUP_1D = 64; // scatter pass: 64 threads/workgroup over the 1D work list

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
    // Sun orthographic view-projection (orthoZO, z in [0,1]) — SAME matrix the sunShadow depth
    // pass uses. Projects a voxel-center world position into the sun shadow map for the SHADOWED
    // sun injection (only matters when the directional sun is enabled; sun.w == 0 otherwise).
    sunViewProj: uC("uSunViewProj", `mat4x4<f32>`),
    // Scatter dispatch description: .x = uTotal (total work items = prefix sum of all AABB
    // voxel counts), .y = uDispatchWidth (threads per workgroup-grid row = dispatchX *
    // WORKGROUP_1D), .z/.w spare. The flat work index is g = gid.y*uDispatchWidth + gid.x.
    dispatch: uC("uDispatch", `vec4<i32>`),
    // Scatter CLASS for this dispatch: 0 = OCCLUDERS (material.x == 0), 1 = EMITTERS. The scatter
    // runs TWICE — emitters into voxelEmission FIRST, then occluders into voxelRadiance mip 0 —
    // in two encoder-barriered passes writing DIFFERENT volumes, so the two classes never race
    // for a shared voxel: the occluder pass reads the emitter volume and MERGES (sum rgb, max a)
    // instead of one class overwriting the other. See main().
    pass: uC("uPass", `u32`),

    // ---- group 1 : per-instance scene storage (StorageRead => @group(1)) ----
    // SAME names/types as sdf.shader.ts / the former gather, so the shared sceneSDF
    // helpers (which read uKind/uValues/uRoundness by global name) see the live scene.
    // Bound to sceneInstances.* GPUVariables in THIS declaration order (bindings 0..5).
    transform: sceneBuf("uTransform", `array<mat4x4<f32>, ${MAX_INSTANCE_COUNT}>`),
    kind: sceneBuf("uKind", `array<u32, ${MAX_INSTANCE_COUNT}>`),
    values: sceneBuf("uValues", `array<f32, ${MAX_INSTANCE_COUNT * 8}>`),
    roundness: sceneBuf("uRoundness", `array<f32, ${MAX_INSTANCE_COUNT}>`),
    color: sceneBuf("uColor", `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`),
    material: sceneBuf("uMaterial", `array<vec4<f32>, ${MAX_INSTANCE_COUNT}>`),
    // Per-instance AABB voxel box (built on the CPU each frame; bindings 6..7 — declared
    // AFTER the 6 scene buffers so their binding numbers are preserved).
    // .xyz = voxel box MIN (vx0,vy0,vz0), .w = prefix start (monotonic non-decreasing).
    aabbMin: sceneBuf("uAabbMin", `array<vec4<i32>, ${MAX_INSTANCE_COUNT}>`),
    // .xyz = voxel box DIMS (nx,ny,nz), .w = n = nx*ny*nz.
    aabbDim: sceneBuf("uAabbDim", `array<vec4<i32>, ${MAX_INSTANCE_COUNT}>`),

    // ---- group 2 : voxel output (StorageTexture, write-only) ----
    // The scatter's write TARGET, bound per pass: the emitter pass (uPass=1) writes the
    // voxelEmission volume, the occluder pass (uPass=0) writes voxelRadiance mip 0 (merging
    // voxelEmission in — it reads it via emissionRead below). The clear pass zeroes whatever
    // is bound (voxelEmission; mip 0 needs no clear — the emission→mip0 copy overwrites it fully).
    voxelTarget: new VariableMeta(
      "voxelTarget",
      VariableKind.StorageTexture,
      `texture_storage_3d<rgba16float, write>`,
      {
        visibility: GPUShaderStage.COMPUTE,
        viewDimension: "3d",
        storageTextureFormat: "rgba16float",
        storageTextureAccess: "write-only",
      },
    ),

    // ---- group 0 : sun shadow map (Texture => @group(0)), COMPUTE-visible ----
    // Sun-POV depth (depth32float from the sunShadow pass). Sampled (single tap, nearest,
    // no PCF) to inject SHADOWED sun into the volume. Read via textureLoad (integer coords
    // + i32 LOD 0 — a depth texture cannot use a filtering sampler).
    shadowMap: new VariableMeta("shadowMap", VariableKind.Texture, `texture_depth_2d`, {
      visibility: GPUShaderStage.COMPUTE,
      textureSampleType: "depth",
    }),

    // ---- group 0 : emitter volume read (Texture), COMPUTE-visible ----
    // voxelEmission, textureLoad-ed by the OCCLUDER pass to merge the emitter contribution into
    // voxelRadiance mip 0 where the two classes share a voxel. The clear/emitter passes WRITE
    // voxelEmission (a texture cannot be sampled and storage-written in one pass), so they bind
    // a 1×1×1 dummy here; the uPass==0 gate in main() keeps the dummy read out of those passes.
    emissionRead: new VariableMeta("emissionRead", VariableKind.Texture, `texture_3d<f32>`, {
      visibility: GPUShaderStage.COMPUTE,
      viewDimension: "3d",
    }),
  },
  {},
  // language=WGSL
  wgsl /* wgsl */ `
const SQRT3: f32 = 1.7320508;

// Shared local-SDF helpers (instance_rot, sd_*, sd_2d_for_kind, extrude, sd_shape3d, sd_normal3d).
// They read uKind/uValues/uRoundness by global name, declared in group 1 with identical types.
${sceneSDF}

// Emission of an instance. material.x encodes intensity; SIGN is a flag, so brightness
// is abs(intensity). intensity == 0 is a pure occluder.
fn emission_of(instance: u32) -> vec3<f32> {
  let intensity = uMaterial[instance].x;
  if (intensity != 0.0) {
    return uColor[instance].rgb * abs(intensity);
  }
  return vec3<f32>(0.0);
}

// Single-tap (nearest, no PCF) sun shadow lookup: project the voxel-center world point P
// (pushed off the surface along N to fight acne) into the sun map and compare, so the sun
// injected into the volume is SHADOWED. Out-of-frustum / past near-far -> lit (level-1 voxels
// beyond the sun frustum — it is fitted to the camera view ∩ the L0 box — thus inject UNSHADOWED
// sun; acceptable for the coarse far field). textureLoad: i32 coords + i32 LOD 0 (a depth
// texture cannot use a filtering sampler). cell = the OWNING LEVEL's cellSize, so the anti-acne
// offset scales with the voxel actually being injected.
fn sun_vis_vox(P: vec3<f32>, N: vec3<f32>, cell: f32) -> f32 {
  let Po = P + N * (1.5 * cell);
  let ls = uSunViewProj * vec4<f32>(Po, 1.0);
  let ndc = ls.xyz / ls.w;
  var uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
  uv.y = 1.0 - uv.y;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  let dim = vec2<i32>(textureDimensions(shadowMap, 0));
  let c = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
  let s = textureLoad(shadowMap, c, 0);
  let bias = 0.002;
  return select(0.0, 1.0, ndc.z <= s + bias);
}

// CLEAR — one thread per voxel over the FULL grid. Zeroes the bound target volume so the
// scatter pass (which writes ONLY solid voxels) starts from an empty volume.
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP}, ${WORKGROUP})
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let coord = vec3<i32>(gid);
  // Dispatch is ceil-rounded; drop threads past the grid bounds.
  if (coord.x >= uGridDims.x || coord.y >= uGridDims.y || coord.z >= uGridDims.z) {
    return;
  }
  textureStore(voxelTarget, coord, vec4<f32>(0.0));
}

// SCATTER — one thread per (instance, voxel-in-its-AABB) pair. The thread:
//   1) flattens its 2D workgroup-grid id into a global work index g,
//   2) binary-searches the per-instance prefix start to find the owning instance,
//   3) decodes the local offset into a voxel coord inside that instance's AABB box,
//   4) evaluates ONLY that instance's local SDF, and writes the voxel if it is solid.
@compute @workgroup_size(${WORKGROUP_1D}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let uTotal = uDispatch.x;
  let uDispatchWidth = uDispatch.y;
  // Flatten the 2D thread grid to a 1D work index (the dispatch is 2D over workgroups to
  // dodge the 65535 per-dimension workgroup limit). uDispatchWidth = the X extent IN THREADS.
  let g = i32(gid.y) * uDispatchWidth + i32(gid.x);
  if (g >= uTotal) {
    return;
  }

  // Binary search the prefix start (= uAabbMin[k].w, monotonic non-decreasing): find the
  // largest ins with start[ins] <= g (standard upper_bound - 1). Empty ranges (n == 0)
  // share their successor's start and are skipped naturally — g never lands inside one.
  var lo: i32 = 0;
  var hi: i32 = i32(uInstanceCount);
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2;
    if (uAabbMin[mid].w <= g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  let ins = u32(lo - 1);

  // CLASS SPLIT — the two classes write DIFFERENT volumes, so neither can overwrite the other
  // (the old "emitter-wins" write-order hack is gone). uPass 1 = emitters scatter into
  // voxelEmission; then the CPU copies voxelEmission → voxelRadiance mip 0 (covers emitter-only
  // voxels); then uPass 0 = occluders (material.x == 0) scatter into voxelRadiance mip 0, ADDING
  // the emitter rgb read back from voxelEmission at their own voxel — a shared voxel ends up
  // with BOTH the occluder's sun-lit surface and the emitter light (sum rgb, max coverage),
  // deterministically, every frame. Early-out BEFORE the SDF eval so off-class threads cost only
  // the binary search above. (Within one class, AABB overlap is still last-writer-wins.)
  let isEmitter = uMaterial[ins].x != 0.0;
  if (isEmitter != (uPass == 1u)) {
    return;
  }

  // Local offset within this instance's voxel box, decoded to (lx,ly,lz).
  let dim = uAabbDim[ins];
  let nx = dim.x;
  let ny = dim.y;
  let local = g - uAabbMin[ins].w;
  let lx = local % nx;
  let ly = (local / nx) % ny;
  let lz = local / (nx * ny);

  // Voxel coord (already clamped to [0,dim) on the CPU side) + its world-space center.
  let coord = uAabbMin[ins].xyz + vec3<i32>(lx, ly, lz);
  let cellSize = uGridOrigin.w;
  let world = uGridOrigin.xyz + (vec3<f32>(coord) + vec3<f32>(0.5)) * cellSize;

  // Evaluate ONLY instance ins: world -> local (subtract center, inverse rotation via
  // transpose), then its LOCAL sd_shape3d. (No loop over the other instances — the whole point.)
  let tr = uTransform[ins];
  let center = vec3<f32>(tr[3].x, tr[3].y, tr[3].z);
  let Rm = instance_rot(tr);
  let s = instance_scale(tr);
  let rel = world - center;
  let lp = (transpose(Rm) * rel) / s;
  let d = sd_shape3d(lp, ins);

  // Conservative "iso-surface crosses this voxel": within half the voxel diagonal. NOT solid
  // -> write nothing (clear already zeroed it; writing zero here would clobber a DIFFERENT
  // instance that legitimately filled this voxel via last-writer-wins).
  if (d > cellSize * 0.5 * SQRT3) {
    return;
  }

  // LOCAL normal of THIS instance only, rotated back to world.
  let nLocal = sd_normal3d(lp, ins);
  let N = Rm * nLocal;

  // voxelRADIANCE (the only volume the cone samples, via its mip pyramid) uses ANTI-ALIASED
  // coverage: a linear band across the iso-surface (1 deep inside -> 0.5 at the centre -> 0 at
  // the outer extent). Stored PREMULTIPLIED (rgb*coverage, a=coverage) so the premultiplied mip
  // pyramid + cone over-operator stay consistent AND a MOVING object's edge voxels fade smoothly
  // instead of popping 0<->1 each frame (kills the voxel-grid shimmer when light passes near).
  let coverage = clamp(0.5 - d / (cellSize * SQRT3), 0.0, 1.0);

  let albedo = uColor[ins].rgb;
  let L = uSun.xyz;
  let ndl = max(dot(N, L), 0.0);
  // The directional sun is injected SHADOWED into the volume: vis from a single-tap sun shadow
  // lookup, so the cone-GI 'indirect' carries an already-shadowed sun. (Only matters when the
  // sun is enabled; uSun.w == 0 -> the whole direct term is 0.)
  let vis = sun_vis_vox(world, N, cellSize);
  let direct = albedo * (ndl * uSun.w) * uSunColor.rgb * vis;
  let emission = emission_of(ins);
  let radiance = direct + emission;

  // radiance is stored PREMULTIPLIED (rgb*coverage, a=coverage) so the mip pyramid + cone
  // over-operator stay consistent. The occluder pass MERGES the emitter volume (already copied
  // into mip 0 for emitter-only voxels) instead of overwriting it: rgb adds, coverage maxes —
  // both stay premultiplied-consistent. The uPass gate also keeps the emitter pass (whose
  // emissionRead is a dummy — it storage-writes the real voxelEmission) from reading garbage.
  if (uPass == 0u) {
    let e = textureLoad(emissionRead, coord, 0);
    textureStore(voxelTarget, coord, vec4<f32>(radiance * coverage + e.rgb, max(coverage, e.a)));
  } else {
    textureStore(voxelTarget, coord, vec4<f32>(radiance * coverage, coverage));
  }
}
`,
);
