import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP, WORKGROUP_1D } from "./voxelize.shader.ts";
import type { SceneInstances } from "../../../SDFSystem/createDrawShapeSystem.ts";
import { SunLight } from "../../../SunLight.ts";
import { buildVoxelAABBs } from "./voxelizeCpu.ts";
import type { createSunShadowSystem } from "../1_sunShadow/sunShadowSystem.ts";

// Grid box (min corner + cellSize + per-axis voxel dims). The AABB build + clear dispatch read it.
export type VoxelizeGridBox = {
  originX: number;
  originY: number;
  originZ: number;
  cellSize: number;
  dimX: number;
  dimY: number;
  dimZ: number;
};

// VOXELIZE cluster: fills the 3D radiance volume (mip 0) from the SDF scene each frame. Owns the
// voxelize compute shader (two pipelines: `clear` zeroes the bound volume, `main` scatters per-shape),
// the two uPass buffers (occluder vs emitter scatter), the group-0/1 bind groups (grid uniforms +
// scene-instance buffers + the sun shadow map), and the per-frame CPU AABB / dispatch scratch. The
// group-0/2 texture bindings + the clear dispatch dims are (re)bound by rebindGrid() when the grid
// is (re)built.
//
// Frame sequence (each step a separate encoder-barriered pass — see voxelize()):
//   1. clear voxelEmission
//   2. EMITTER scatter (uPass=1) → voxelEmission
//   3. copyTextureToTexture voxelEmission → voxelRadiance mip 0 (covers emitter-only voxels;
//      doubles as the mip-0 clear — every voxel is overwritten)
//   4. OCCLUDER scatter (uPass=0) → voxelRadiance mip 0, MERGING voxelEmission (sum rgb, max a)
// The two classes never write the same volume, so the old "emitter-wins" write-order hack (and
// the light it dropped on shared voxels) is gone.
export function createVoxelizeSystem({
  device,
  sceneInstances,
  getGridBox,
  sun,
}: {
  device: GPUDevice;
  sceneInstances: SceneInstances;
  // Grid box (origin + cellSize + voxel dims), read live each frame for the CPU AABB build.
  getGridBox: () => VoxelizeGridBox;
  // The sun shadow sub-system: voxelize binds its depth view (shadowed sun injection) and, at the
  // head of run(), uploads the sun view-proj matrix it computed (must match the map it rendered).
  sun: ReturnType<typeof createSunShadowSystem>;
}) {
  // Two compute pipelines from the one shader: `clear` zeroes the full volume (one thread per
  // voxel), `main` is the per-shape scatter (one thread per (instance, voxel-in-AABB) pair).
  // They share one pipeline layout (groups 0/1/2 — clear binds the same groups, harmless since
  // it does not read the aabb* / scene buffers).
  const voxShader = new GPUShader(voxelizeMeta);
  const voxClearPipeline = voxShader.getComputePipeline(device, "clear");
  const voxPipeline = voxShader.getComputePipeline(device, "main");

  // Sun shadow map: the sun-POV depth texture, sampled to shadow the injected directional sun.
  const sunDepthView = sun.getDepthView();

  // Two tiny constant uPass buffers (0 = occluders, 1 = emitters), uploaded ONCE. The scatter is
  // dispatched twice — once with each — into DIFFERENT target volumes (emitters → voxelEmission,
  // occluders → voxelRadiance mip 0 with the emission merged in), so the classes never race.
  // Two SEPARATE buffers (not one re-uploaded between passes) because both dispatches are encoded
  // before the encoder is submitted: a mid-encode writeBuffer would apply to BOTH passes, not one.
  const passBufOcc = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const passBufEmit = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  {
    const p = new Uint32Array(4);
    p[0] = 0;
    device.queue.writeBuffer(passBufOcc, 0, p);
    p[0] = 1;
    device.queue.writeBuffer(passBufEmit, 0, p);
  }

  // 1×1×1 dummy for the emissionRead binding in the passes that WRITE voxelEmission (clear +
  // emitter scatter) — a texture cannot be sampled and storage-written in the same pass, and the
  // uPass==1 branch never reads it anyway. Never written; reads as zero.
  const dummyEmissionView = device
    .createTexture({
      size: [1, 1, 1],
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    })
    .createView({ dimension: "3d" });

  // Group 0 (voxelize) = grid uniforms + the sun shadow map + the emissionRead texture; Group 1 =
  // the scene-instance buffers (stable → built ONCE). Scene buffers are bound at the VOXELIZE
  // meta's binding numbers (NOT sceneInstances.X.getBindGroupEntry(), which carries the DRAW
  // shader's bindings). Two group-0 variants: occluder (uPass=0, reads the REAL voxelEmission for
  // the merge) vs emitter/clear (uPass=1, dummy read). Rebuilt by rebindGrid — the emission view
  // is grid-sized.
  const makeVoxGroup0 = (passBuf: GPUBuffer, emissionReadView: GPUTextureView) =>
    device.createBindGroup({
      layout: voxPipeline.getBindGroupLayout(0),
      entries: [
        voxShader.uniforms.gridOrigin.getBindGroupEntry(device),
        voxShader.uniforms.gridDims.getBindGroupEntry(device),
        voxShader.uniforms.instanceCount.getBindGroupEntry(device),
        voxShader.uniforms.sun.getBindGroupEntry(device),
        voxShader.uniforms.sunColor.getBindGroupEntry(device),
        voxShader.uniforms.sunViewProj.getBindGroupEntry(device),
        voxShader.uniforms.dispatch.getBindGroupEntry(device),
        // Sun shadow map: the sun-POV depth texture, sampled to shadow the injected directional sun.
        { binding: voxelizeMeta.uniforms.shadowMap.binding, resource: sunDepthView },
        { binding: voxelizeMeta.uniforms.pass.binding, resource: { buffer: passBuf } },
        { binding: voxelizeMeta.uniforms.emissionRead.binding, resource: emissionReadView },
      ],
    });
  const voxGroup1 = device.createBindGroup({
    layout: voxPipeline.getBindGroupLayout(1),
    entries: [
      {
        binding: voxelizeMeta.uniforms.transform.binding,
        resource: { buffer: sceneInstances.transform.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.kind.binding,
        resource: { buffer: sceneInstances.kind.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.values.binding,
        resource: { buffer: sceneInstances.values.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.roundness.binding,
        resource: { buffer: sceneInstances.roundness.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.color.binding,
        resource: { buffer: sceneInstances.color.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.material.binding,
        resource: { buffer: sceneInstances.material.getGPUBuffer(device) },
      },
      // Per-instance voxel-AABB boxes + prefix sum, built on the CPU each frame and uploaded to
      // these two new StorageRead buffers (their own GPUVariables — sized MAX_INSTANCE_COUNT*4
      // i32, STORAGE | COPY_DST). Bound at the voxelize meta's new binding numbers (7, 8).
      voxShader.uniforms.aabbMin.getBindGroupEntry(device),
      voxShader.uniforms.aabbDim.getBindGroupEntry(device),
    ],
  });

  // --- Scratch typed arrays for uniform uploads. ---
  const instanceCountArr = getTypeTypedArray(voxelizeMeta.uniforms.instanceCount.type); // Uint32Array(1)
  const sunArr = getTypeTypedArray(voxelizeMeta.uniforms.sun.type); // Float32Array(4)
  const sunColorArr = getTypeTypedArray(voxelizeMeta.uniforms.sunColor.type); // Float32Array(4)
  // The sun view-proj voxelize samples the shadow map with (for the shadowed sun injection).
  const voxSunViewProjArr = getTypeTypedArray(voxelizeMeta.uniforms.sunViewProj.type); // Float32Array(16)
  // Scatter dispatch: .x = total work items, .y = threads per workgroup-grid row, .z/.w spare.
  const dispatchArr = getTypeTypedArray(voxelizeMeta.uniforms.dispatch.type); // Int32Array(4)
  // Per-instance AABB scratch (allocated ONCE — never per frame). Packed vec4<i32> per instance:
  // aabbMinArr[k*4 + 0..2] = voxel box min, +3 = prefix start; aabbDimArr[k*4 + 0..2] = dims, +3 = n.
  const aabbMinArr = getTypeTypedArray(voxelizeMeta.uniforms.aabbMin.type) as Int32Array; // Int32Array(MAX*4)
  const aabbDimArr = getTypeTypedArray(voxelizeMeta.uniforms.aabbDim.type) as Int32Array; // Int32Array(MAX*4)
  // Grid uniform scratch (written to voxShader in rebindGrid; content mirrors buildGrid's).
  const originArr = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)

  // --- Grid-dependent state (rebuilt by rebindGrid). ---
  // Group-0 variants: occluder pass (uPass=0 + real emission read) vs emitter/clear passes
  // (uPass=1 + dummy read). Group-2 variants: the write target — voxelRadiance mip 0 (occluder)
  // vs voxelEmission (emitter + clear).
  let voxGroup0Occ: GPUBindGroup;
  let voxGroup0Emit: GPUBindGroup;
  let voxGroup2Radiance: GPUBindGroup;
  let voxGroup2Emission: GPUBindGroup;
  // Texture refs + dims for the per-frame voxelEmission → voxelRadiance mip-0 copy.
  let radianceTex: GPUTexture;
  let emissionTex: GPUTexture;
  let gridDimX = 0;
  let gridDimY = 0;
  let gridDimZ = 0;
  let dispatchX = 0;
  let dispatchY = 0;
  let dispatchZ = 0;
  // Scatter dispatch (rebuilt every frame from the prefix-sum total in run()).
  let scatterTotal = 0;
  let scatterDispatchX = 0;
  let scatterDispatchY = 0;

  // Rebind the two storage targets + the emission merge read, and refresh the clear dispatch dims
  // and grid uniforms for the (re)built grid. Called from buildGrid after the volumes are recreated.
  function rebindGrid(
    voxelRadiance: GPUTexture,
    voxelEmission: GPUTexture,
    gridBox: VoxelizeGridBox,
  ) {
    const { originX, originY, originZ, cellSize, dimX, dimY, dimZ } = gridBox;

    radianceTex = voxelRadiance;
    emissionTex = voxelEmission;

    const emissionReadView = voxelEmission.createView({ dimension: "3d" });
    voxGroup0Occ = makeVoxGroup0(passBufOcc, emissionReadView);
    voxGroup0Emit = makeVoxGroup0(passBufEmit, dummyEmissionView);

    // Group 2 (voxelize) = the write target (write-only storage, dimension 3d). voxelRadiance has
    // a mip pyramid; a storage view MUST span exactly one mip → bind mip 0 only (the voxelize pass
    // writes level 0; voxelMip builds the rest). voxelEmission is single-mip.
    const makeVoxGroup2 = (view: GPUTextureView) =>
      device.createBindGroup({
        layout: voxPipeline.getBindGroupLayout(2),
        entries: [{ binding: voxelizeMeta.uniforms.voxelTarget.binding, resource: view }],
      });
    voxGroup2Radiance = makeVoxGroup2(
      voxelRadiance.createView({ dimension: "3d", baseMipLevel: 0, mipLevelCount: 1 }),
    );
    voxGroup2Emission = makeVoxGroup2(voxelEmission.createView({ dimension: "3d" }));

    // Grid uniforms.
    originArr[0] = originX;
    originArr[1] = originY;
    originArr[2] = originZ;
    originArr[3] = cellSize;
    dimsArr[0] = dimX;
    dimsArr[1] = dimY;
    dimsArr[2] = dimZ;
    dimsArr[3] = 0;
    device.queue.writeBuffer(voxShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(voxShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);

    gridDimX = dimX;
    gridDimY = dimY;
    gridDimZ = dimZ;
    dispatchX = Math.ceil(dimX / WORKGROUP);
    dispatchY = Math.ceil(dimY / WORKGROUP);
    dispatchZ = Math.ceil(dimZ / WORKGROUP);
  }

  // Re-voxelize the scene into the 3D textures (run before debug()/the GI gather). Runs after
  // sunDepth (sun.render) every frame; its shadow map + view-proj matrix are current.
  function voxelize(encoder: GPUCommandEncoder) {
    // uSunViewProj: the matrix that rendered THIS frame's shadow map (sunDepth ran just before), so
    // the shadowed-sun injection samples the matching matrix. Uploaded at the head of the pass; all
    // queue writes land before the dispatch, so it is byte-identical to an earlier upload.
    const sunViewProj = sun.getSunViewProj();
    voxSunViewProjArr.set(sunViewProj as Float32Array);
    device.queue.writeBuffer(
      voxShader.uniforms.sunViewProj.getGPUBuffer(device),
      0,
      voxSunViewProjArr,
    );

    instanceCountArr[0] = sceneInstances.instanceCount;
    device.queue.writeBuffer(
      voxShader.uniforms.instanceCount.getGPUBuffer(device),
      0,
      instanceCountArr,
    );

    // Directional sun, recomputed each frame from the SunLight singleton. .xyz = world
    // dir TOWARD the sun (azimuth + elevation), .w = effective intensity (0 = disabled).
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    sunArr[0] = Math.cos(a) * ce;
    sunArr[1] = Math.sin(a) * ce;
    sunArr[2] = Math.sin(e);
    sunArr[3] = SunLight.enabled ? SunLight.intensity : 0;
    device.queue.writeBuffer(voxShader.uniforms.sun.getGPUBuffer(device), 0, sunArr);
    sunColorArr[0] = SunLight.color[0];
    sunColorArr[1] = SunLight.color[1];
    sunColorArr[2] = SunLight.color[2];
    device.queue.writeBuffer(voxShader.uniforms.sunColor.getGPUBuffer(device), 0, sunColorArr);

    // Build per-instance voxel AABBs + the scatter prefix-sum work list on the CPU. The scene CPU
    // mirrors are filled by prepare() (runs before voxelize() each frame), so they are current.
    // See voxelizeCpu.buildVoxelAABBs.
    const { originX, originY, originZ, cellSize, dimX, dimY, dimZ } = getGridBox();
    const n = sceneInstances.instanceCount;
    scatterTotal = buildVoxelAABBs(
      sceneInstances,
      { originX, originY, originZ, cellSize, dimX, dimY, dimZ },
      aabbMinArr,
      aabbDimArr,
    );

    // Upload the AABB lists. Only the live n entries matter (the binary search bound is
    // uInstanceCount = n), so upload exactly n*4 i32 elements instead of the whole MAX-sized buffer.
    device.queue.writeBuffer(
      voxShader.uniforms.aabbMin.getGPUBuffer(device),
      0,
      aabbMinArr,
      0,
      n * 4,
    );
    device.queue.writeBuffer(
      voxShader.uniforms.aabbDim.getGPUBuffer(device),
      0,
      aabbDimArr,
      0,
      n * 4,
    );

    // Scatter dispatch sizing — 2D over workgroups to dodge the 65535 per-dim workgroup cap.
    const wgTotal = Math.ceil(scatterTotal / WORKGROUP_1D);
    scatterDispatchX = Math.min(wgTotal, 65535);
    scatterDispatchY = scatterDispatchX > 0 ? Math.ceil(wgTotal / scatterDispatchX) : 0;
    // uDispatch: .x = total work items, .y = threads per workgroup-grid row (= dispatchX * WG).
    dispatchArr[0] = scatterTotal;
    dispatchArr[1] = scatterDispatchX * WORKGROUP_1D;
    dispatchArr[2] = 0;
    dispatchArr[3] = 0;
    device.queue.writeBuffer(voxShader.uniforms.dispatch.getGPUBuffer(device), 0, dispatchArr);

    // Four encoder-barriered steps (dispatches within ONE pass are NOT synchronized — every
    // consumer must be a separate pass/copy so it sees the producer's writes):
    //   1. CLEAR voxelEmission (full grid) — the emitter scatter only writes solid voxels.
    //   2. EMITTER scatter (uPass=1) → voxelEmission.
    //   3. COPY voxelEmission → voxelRadiance mip 0 — lands the emitter-only voxels AND doubles
    //      as the mip-0 clear (every voxel is overwritten, empty ones with zero).
    //   4. OCCLUDER scatter (uPass=0) → voxelRadiance mip 0, MERGING the emission it reads back
    //      (sum rgb, max coverage) — a voxel shared by both classes keeps BOTH contributions,
    //      deterministically every frame (the old emitter-wins overwrite dropped the occluder's).
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(voxClearPipeline);
    // Emitter group-0 variant: clear WRITES voxelEmission, so it must bind the dummy read.
    clearPass.setBindGroup(0, voxGroup0Emit);
    clearPass.setBindGroup(1, voxGroup1);
    clearPass.setBindGroup(2, voxGroup2Emission);
    clearPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    clearPass.end();

    if (scatterTotal > 0) {
      // The two scatter passes run over the SAME work list: each invocation binary-searches its
      // owning instance and early-outs unless it belongs to this pass's class, so the SDF-eval
      // work is split (not duplicated).
      const scatterEmit = encoder.beginComputePass();
      scatterEmit.setPipeline(voxPipeline);
      scatterEmit.setBindGroup(0, voxGroup0Emit);
      scatterEmit.setBindGroup(1, voxGroup1);
      scatterEmit.setBindGroup(2, voxGroup2Emission);
      scatterEmit.dispatchWorkgroups(scatterDispatchX, scatterDispatchY, 1);
      scatterEmit.end();
    }

    encoder.copyTextureToTexture(
      { texture: emissionTex },
      { texture: radianceTex, mipLevel: 0 },
      [gridDimX, gridDimY, gridDimZ],
    );

    if (scatterTotal > 0) {
      const scatterOcc = encoder.beginComputePass();
      scatterOcc.setPipeline(voxPipeline);
      scatterOcc.setBindGroup(0, voxGroup0Occ);
      scatterOcc.setBindGroup(1, voxGroup1);
      scatterOcc.setBindGroup(2, voxGroup2Radiance);
      scatterOcc.dispatchWorkgroups(scatterDispatchX, scatterDispatchY, 1);
      scatterOcc.end();
    }
  }

  return {
    voxelize,
    rebindGrid,
  };
}
