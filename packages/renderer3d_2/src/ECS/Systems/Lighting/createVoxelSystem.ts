import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP } from "./voxelize.shader.ts";
import { shaderMeta as debugMeta } from "./voxelDebug.shader.ts";
import { shaderMeta as coneMeta } from "./voxelCone.shader.ts";
import { shaderMeta as compositeMeta } from "./voxelComposite.shader.ts";
import { shaderMeta as mipMeta, WORKGROUP as MIP_WG } from "./voxelMip.shader.ts";
import {
  createVoxelTextures,
  DEFAULT_VOXEL_GRID,
  voxelMipLevelCount,
  type VoxelGridConfig,
  type VoxelTextures,
} from "./voxelResources.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";
import { SunLight } from "../SunLight.ts";

export type VoxelParams = {
  // Ambient floor added to the debug Lambert shade.
  ambient: number;
};

export type VoxelConeParams = {
  normalBias: number; // extra lift of the cone origin off the surface
  maxDist: number; // cone reach (world units)
  aperture: number; // tan(halfAngle) — cone half-angle (~0.577 = 60° full angle)
  giStrength: number; // multiplier on the gathered diffuse-cone radiance
};

export type VoxelCompositeParams = {
  ambient: number; // ambient floor (scaled by the cone's AO term)
};

// Voxel scene system: voxelize() fills the 3D albedo/emission/radiance textures from the SDF
// scene each frame; mips() builds the radiance pyramid; cone() gathers indirect light (6-cone
// VCT) and composite() produces the final lit image. debug() raymarches the voxels for
// inspection (see docs/voxel-cone-tracing-impl.md).
//
// GRANULARITY: the world box (origin + extent) is FIXED; cellSize controls voxel size
// (and thus per-axis dims = round(extent/cellSize)) — the "graininess" knob. Smaller
// cellSize = finer voxels = more of them. setCellSize() rebuilds the textures + the two
// texture-referencing bind groups; the canvas-sized output texture is independent of it.
export function createVoxelSystem({
  device,
  canvas,
  sceneInstances,
  depthTexture,
  normalTexture,
  albedoTexture,
  emissionTexture,
  grid = DEFAULT_VOXEL_GRID,
}: {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  sceneInstances: SceneInstances;
  // G-buffer (the SDF draw pass output): reverse-Z depth + world normal + albedo + per-pixel
  // self-emission. The cone pass reads depth + normal to reconstruct P + N; the composite
  // reads albedo + emission.
  depthTexture: GPUTexture;
  normalTexture: GPUTexture;
  albedoTexture: GPUTexture;
  emissionTexture: GPUTexture;
  grid?: VoxelGridConfig;
}) {
  const params: VoxelParams = { ambient: 0.12 };
  const coneParams: VoxelConeParams = {
    normalBias: 0,
    maxDist: 24,
    aperture: 0.577,
    giStrength: 1,
  };
  // giStrength stays in coneParams (baked into the cone's rgb); the composite only adds the
  // ambient floor.
  const compositeParams: VoxelCompositeParams = { ambient: 0.25 };

  // G-buffer textures (reassigned by recreate() on canvas resize).
  let gDepth = depthTexture;
  let gNormal = normalTexture;
  let gAlbedo = albedoTexture;
  let gEmission = emissionTexture;

  // Fixed world box (min corner + extent), derived from the initial config. cellSize is
  // the only thing that varies; dims follow from it.
  const originX = grid.originX;
  const originY = grid.originY;
  const originZ = grid.originZ;
  const extentX = grid.dimX * grid.cellSize;
  const extentY = grid.dimY * grid.cellSize;
  const extentZ = grid.dimZ * grid.cellSize;

  // ===== Shaders + pipelines (created once; only textures/bind groups rebuild). =====
  const voxShader = new GPUShader(voxelizeMeta);
  const voxPipeline = voxShader.getComputePipeline(device, "main");
  const debugShader = new GPUShader(debugMeta);
  const debugPipeline = debugShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "bgra8unorm", // matches the output texture + frame.renderTexture
    withBlending: false,
  });
  // VCT cone GI — 6-cone diffuse hemisphere gather over the G-buffer → HALF-res HDR target
  // (composite bilinear-upsamples to full res; the heavy cone work runs at ¼ the pixels).
  const coneShader = new GPUShader(coneMeta);
  const conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // VCT composite (Layer 4): final = albedo·(ambient·AO + directSun + indirect) + emission.
  const compositeShader = new GPUShader(compositeMeta);
  const compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // Voxel-radiance mip-pyramid downsample (one dispatch per level).
  const mipShader = new GPUShader(mipMeta);
  const mipPipeline = mipShader.getComputePipeline(device, "main");
  // The mip shader uses groups 0 (uniform+src) and 2 (dst storage) with NOTHING in group 1,
  // so its pipeline layout has an empty layout at index 1. Bind a matching empty group there
  // each dispatch, so strict implementations that require every layout index to be set are
  // satisfied. (The layout object is the same one the pipeline layout reflects.)
  const mipEmptyGroup1 = device.createBindGroup({
    layout: mipShader.createBindGroupLayout(device, 1),
    entries: [],
  });

  // Filtering sampler for the LOD debug (textureSampleLevel over the rgba16float pyramid).
  const voxelSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
  });

  // Group 0 (voxelize) = grid uniforms; Group 1 = the 7 scene-instance buffers. Both
  // reference stable buffers (uniform + draw system's GPUVariables) → built ONCE. Scene
  // buffers are bound at the VOXELIZE meta's binding numbers (NOT
  // sceneInstances.X.getBindGroupEntry(), which carries the DRAW shader's bindings).
  const voxGroup0 = device.createBindGroup({
    layout: voxPipeline.getBindGroupLayout(0),
    entries: [
      voxShader.uniforms.gridOrigin.getBindGroupEntry(device),
      voxShader.uniforms.gridDims.getBindGroupEntry(device),
      voxShader.uniforms.instanceCount.getBindGroupEntry(device),
      voxShader.uniforms.sun.getBindGroupEntry(device),
      voxShader.uniforms.sunColor.getBindGroupEntry(device),
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
        binding: voxelizeMeta.uniforms.heights.binding,
        resource: { buffer: sceneInstances.heights.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.color.binding,
        resource: { buffer: sceneInstances.color.getGPUBuffer(device) },
      },
      {
        binding: voxelizeMeta.uniforms.material.binding,
        resource: { buffer: sceneInstances.material.getGPUBuffer(device) },
      },
    ],
  });

  // --- Scratch typed arrays for uniform uploads. ---
  const originArr = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)
  const instanceCountArr = getTypeTypedArray(voxelizeMeta.uniforms.instanceCount.type); // Uint32Array(1)
  const sunArr = getTypeTypedArray(voxelizeMeta.uniforms.sun.type); // Float32Array(4)
  const sunColorArr = getTypeTypedArray(voxelizeMeta.uniforms.sunColor.type); // Float32Array(4)
  const paramsArr = getTypeTypedArray(debugMeta.uniforms.params.type); // Float32Array(4)
  const invViewProj = mat4.create();
  const invArr = getTypeTypedArray(debugMeta.uniforms.invViewProj.type); // Float32Array(16)
  // Cone scratch.
  const coneParamsArr = getTypeTypedArray(coneMeta.uniforms.params.type); // Float32Array(4)
  const coneParams2Arr = getTypeTypedArray(coneMeta.uniforms.params2.type); // Float32Array(4)
  const coneInvArr = getTypeTypedArray(coneMeta.uniforms.invViewProj.type); // Float32Array(16)
  // Composite scratch.
  const compParamsArr = getTypeTypedArray(compositeMeta.uniforms.params.type); // Float32Array(4)
  const compParams2Arr = getTypeTypedArray(compositeMeta.uniforms.params2.type); // Float32Array(4)
  const compSunArr = getTypeTypedArray(compositeMeta.uniforms.sun.type); // Float32Array(4)
  const compSunColorArr = getTypeTypedArray(compositeMeta.uniforms.sunColor.type); // Float32Array(4)
  // Mip scratch: .xyz = destination mip dims (re-uploaded per level).
  const mipArr = getTypeTypedArray(mipMeta.uniforms.mip.type); // Int32Array(4)

  // --- Grid state (rebuilt by buildGrid). ---
  let cellSize = grid.cellSize;
  let dimX = grid.dimX;
  let dimY = grid.dimY;
  let dimZ = grid.dimZ;
  let textures: VoxelTextures;
  let voxGroup2: GPUBindGroup;
  let debugGroup0: GPUBindGroup;
  let coneGroup0: GPUBindGroup;
  let compositeGroup0: GPUBindGroup;
  let dispatchX = 0;
  let dispatchY = 0;
  let dispatchZ = 0;

  // Mip-pyramid state (rebuilt by buildGrid). One downsample step per pair of adjacent
  // levels → mipCount-1 steps, indexed 0..mipCount-2. mipGroup0[L]/mipGroup2[L] downsample
  // mip L → mip L+1.
  let mipCount = 1;
  let mipBuf: GPUBuffer[] = [];
  let mipGroup0: GPUBindGroup[] = [];
  let mipGroup2: GPUBindGroup[] = [];

  // (Re)build the Layer-2 cone bind group: uniforms + the G-buffer (depth/normal) + the
  // ALL-mips voxelRadiance view + the shared filtering sampler. Rebuilt whenever the
  // voxelRadiance view changes (grid rebuild) or the G-buffer changes (canvas resize).
  function buildConeGroup() {
    coneGroup0 = device.createBindGroup({
      layout: conePipeline.getBindGroupLayout(0),
      entries: [
        coneShader.uniforms.params.getBindGroupEntry(device),
        coneShader.uniforms.params2.getBindGroupEntry(device),
        coneShader.uniforms.invViewProj.getBindGroupEntry(device),
        coneShader.uniforms.gridOrigin.getBindGroupEntry(device),
        coneShader.uniforms.gridDims.getBindGroupEntry(device),
        { binding: coneMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: coneMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        // ALL-mips sampled view so textureSampleLevel can pick any lod.
        {
          binding: coneMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
        { binding: coneMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
      ],
    });
  }

  // (Re)build the Layer-4 composite bind group: uniforms + the G-buffer (albedo/normal/
  // emission) + the cone output (indirect+AO). Self-emission is now a per-pixel G-buffer
  // target (surface property), so the composite no longer needs depth/invViewProj/grid or
  // the voxelEmission 3D view. Rebuilt when the G-buffer / coneOutput change (resize).
  function buildCompositeGroup() {
    compositeGroup0 = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        compositeShader.uniforms.params.getBindGroupEntry(device),
        compositeShader.uniforms.params2.getBindGroupEntry(device),
        compositeShader.uniforms.sun.getBindGroupEntry(device),
        compositeShader.uniforms.sunColor.getBindGroupEntry(device),
        { binding: compositeMeta.uniforms.albedoTex.binding, resource: gAlbedo.createView() },
        { binding: compositeMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: compositeMeta.uniforms.coneTex.binding, resource: coneOutput.createView() },
        // Linear/clamp sampler (reuse voxelSampler) bilinearly upsamples the half-res cone.
        { binding: compositeMeta.uniforms.coneSampler.binding, resource: voxelSampler },
        {
          binding: compositeMeta.uniforms.emissionTex.binding,
          resource: gEmission.createView(),
        },
      ],
    });
  }

  // (Re)build the voxel textures + the two texture-referencing bind groups for the
  // current cellSize, and upload the grid uniforms to all shaders.
  function buildGrid(newCellSize: number) {
    cellSize = newCellSize;
    dimX = Math.max(1, Math.round(extentX / cellSize));
    dimY = Math.max(1, Math.round(extentY / cellSize));
    dimZ = Math.max(1, Math.round(extentZ / cellSize));

    textures = createVoxelTextures(device, {
      originX,
      originY,
      originZ,
      dimX,
      dimY,
      dimZ,
      cellSize,
    });

    // Group 2 (voxelize) = voxel output storage textures (write-only, dimension 3d).
    // voxelRadiance now has a mip pyramid; a storage view MUST span exactly one mip → bind
    // mip 0 only (the voxelize pass writes level 0; voxelMip builds the rest).
    voxGroup2 = device.createBindGroup({
      layout: voxPipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: voxelizeMeta.uniforms.voxelAlbedo.binding,
          resource: textures.voxelAlbedo.createView({ dimension: "3d" }),
        },
        {
          binding: voxelizeMeta.uniforms.voxelEmission.binding,
          resource: textures.voxelEmission.createView({ dimension: "3d" }),
        },
        {
          binding: voxelizeMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({
            dimension: "3d",
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
      ],
    });

    // Group 0 (debug) = uniforms (stable buffers) + the SAME textures sampled as 3d.
    debugGroup0 = device.createBindGroup({
      layout: debugPipeline.getBindGroupLayout(0),
      entries: [
        debugShader.uniforms.params.getBindGroupEntry(device),
        debugShader.uniforms.invViewProj.getBindGroupEntry(device),
        debugShader.uniforms.gridOrigin.getBindGroupEntry(device),
        debugShader.uniforms.gridDims.getBindGroupEntry(device),
        { binding: debugMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
        {
          binding: debugMeta.uniforms.voxelAlbedo.binding,
          resource: textures.voxelAlbedo.createView({ dimension: "3d" }),
        },
        {
          binding: debugMeta.uniforms.voxelEmission.binding,
          resource: textures.voxelEmission.createView({ dimension: "3d" }),
        },
        // ALL-mips sampled view so textureSampleLevel can pick any lod.
        {
          binding: debugMeta.uniforms.voxelRadiance.binding,
          resource: textures.voxelRadiance.createView({ dimension: "3d" }),
        },
      ],
    });

    // Mip-pyramid downsample groups (mip L → L+1). srcView is a single-mip SAMPLED view of
    // mip L; dstView is a single-mip STORAGE view of mip L+1 (different subresources of the
    // same texture → allowed). Rebuilt here because views/buffers depend on the new dims.
    for (const b of mipBuf) b.destroy();
    mipCount = voxelMipLevelCount(dimX, dimY, dimZ);
    mipBuf = [];
    mipGroup0 = [];
    mipGroup2 = [];
    for (let L = 0; L < mipCount - 1; L++) {
      const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      mipBuf.push(buf);

      const srcView = textures.voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L,
        mipLevelCount: 1,
      });
      const dstView = textures.voxelRadiance.createView({
        dimension: "3d",
        baseMipLevel: L + 1,
        mipLevelCount: 1,
      });

      mipGroup0.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(0),
          entries: [
            { binding: mipMeta.uniforms.mip.binding, resource: { buffer: buf } },
            { binding: mipMeta.uniforms.src.binding, resource: srcView },
          ],
        }),
      );
      mipGroup2.push(
        device.createBindGroup({
          layout: mipPipeline.getBindGroupLayout(2),
          entries: [{ binding: mipMeta.uniforms.dst.binding, resource: dstView }],
        }),
      );

      // Destination dims = mip L+1 dims (halved per axis, floored at 1).
      mipArr[0] = Math.max(1, dimX >> (L + 1));
      mipArr[1] = Math.max(1, dimY >> (L + 1));
      mipArr[2] = Math.max(1, dimZ >> (L + 1));
      mipArr[3] = 0;
      device.queue.writeBuffer(buf, 0, mipArr);
    }

    // Cone bind group references the rebuilt voxelRadiance view + the (stable) G-buffer.
    buildConeGroup();
    // Composite bind group references the rebuilt voxelEmission view + G-buffer + coneOutput.
    buildCompositeGroup();

    // Grid uniforms (shared by all shaders).
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
    device.queue.writeBuffer(debugShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(debugShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    device.queue.writeBuffer(coneShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(coneShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    // Composite no longer reads grid uniforms (self-emission is a per-pixel G-buffer target).

    dispatchX = Math.ceil(dimX / WORKGROUP);
    dispatchY = Math.ceil(dimY / WORKGROUP);
    dispatchZ = Math.ceil(dimZ / WORKGROUP);
  }
  // Canvas-sized debug output (presented). Independent of the voxel grid.
  const createOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "bgra8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let outputTexture = createOutput();

  // HALF-res HDR target for the Layer-2 cone gather (presented as the "cone" source).
  // Half-res for perf (¼ the pixels → ~4× less cone work); the composite bilinear-upsamples
  // it back to full res (indirect light is low-frequency, so that is fine). The render pass
  // viewport IS the texture size, so the cone pass renders at half res automatically.
  const createConeOutput = () =>
    device.createTexture({
      size: [
        Math.max(1, Math.ceil(canvas.width / 2)),
        Math.max(1, Math.ceil(canvas.height / 2)),
        1,
      ],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let coneOutput = createConeOutput();

  // Full-res HDR target for the Layer-4 composite (the final lit image — "final" source).
  const createCompositeOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let compositeOutput = createCompositeOutput();

  // Built last: buildGrid() (and recreate()) call buildCompositeGroup(), which references
  // coneOutput + compositeOutput, so those textures must exist first.
  buildGrid(cellSize);

  // Re-voxelize the scene into the 3D textures (run before debug()/the GI gather).
  function voxelize(encoder: GPUCommandEncoder) {
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

    const pass = encoder.beginComputePass();
    pass.setPipeline(voxPipeline);
    pass.setBindGroup(0, voxGroup0);
    pass.setBindGroup(1, voxGroup1);
    pass.setBindGroup(2, voxGroup2);
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    pass.end();
  }

  // Build the voxelRadiance mip pyramid: one compute pass PER level (passes are ordered/
  // barriered by the encoder, so level L+1 sees level L's writes — dispatches WITHIN a pass
  // are not synchronized, hence one pass each). Must run AFTER voxelize() (level 0 reads
  // mip 0 that voxelize wrote) and in the SAME encoder.
  function mips(encoder: GPUCommandEncoder) {
    for (let L = 0; L < mipCount - 1; L++) {
      const dx = Math.max(1, dimX >> (L + 1));
      const dy = Math.max(1, dimY >> (L + 1));
      const dz = Math.max(1, dimZ >> (L + 1));
      const pass = encoder.beginComputePass();
      pass.setPipeline(mipPipeline);
      pass.setBindGroup(0, mipGroup0[L]);
      pass.setBindGroup(1, mipEmptyGroup1);
      pass.setBindGroup(2, mipGroup2[L]);
      pass.dispatchWorkgroups(
        Math.ceil(dx / MIP_WG),
        Math.ceil(dy / MIP_WG),
        Math.ceil(dz / MIP_WG),
      );
      pass.end();
    }
  }

  // Raymarch the voxel grid into outputTexture. mode: 0 = lit albedo, 1 = stored radiance,
  // 2 = LOD sample of the radiance pyramid at `lod` (requires mips() to have run).
  function debug(encoder: GPUCommandEncoder, mode = 0, lod = 0) {
    paramsArr[0] = params.ambient;
    paramsArr[1] = mode;
    paramsArr[2] = lod;
    device.queue.writeBuffer(debugShader.uniforms.params.getGPUBuffer(device), 0, paramsArr);

    mat4.invert(invViewProj, viewProjMatrix);
    invArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(debugShader.uniforms.invViewProj.getGPUBuffer(device), 0, invArr);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: outputTexture.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(debugPipeline);
    pass.setBindGroup(0, debugGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // VCT cone GI: trace the 6-cone diffuse hemisphere at the per-pixel normal through the
  // voxelRadiance pyramid → coneOutput (HALF-res HDR; composite bilinear-upsamples). MUST run
  // AFTER voxelize() + mips() (it samples the pyramid). Reads the G-buffer (depth + normal).
  function cone(encoder: GPUCommandEncoder) {
    coneParamsArr[0] = coneParams.normalBias;
    coneParamsArr[1] = coneParams.maxDist;
    coneParamsArr[2] = coneParams.aperture;
    coneParamsArr[3] = coneParams.giStrength;
    device.queue.writeBuffer(coneShader.uniforms.params.getGPUBuffer(device), 0, coneParamsArr);

    coneParams2Arr[0] = canvas.width;
    coneParams2Arr[1] = canvas.height;
    coneParams2Arr[2] = 0;
    coneParams2Arr[3] = 0;
    device.queue.writeBuffer(coneShader.uniforms.params2.getGPUBuffer(device), 0, coneParams2Arr);

    mat4.invert(invViewProj, viewProjMatrix);
    coneInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(coneShader.uniforms.invViewProj.getGPUBuffer(device), 0, coneInvArr);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: coneOutput.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(conePipeline);
    pass.setBindGroup(0, coneGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // VCT composite (Layer 4): combine albedo + cone indirect/AO + direct sun + self-emission
  // into the final lit image → compositeOutput (full-res HDR). MUST run AFTER cone() (it reads
  // coneOutput). Reads the G-buffer (albedo/normal/depth) + voxelEmission.
  function composite(encoder: GPUCommandEncoder) {
    compParamsArr[0] = compositeParams.ambient;
    compParamsArr[1] = 0;
    compParamsArr[2] = 0;
    compParamsArr[3] = 0;
    device.queue.writeBuffer(
      compositeShader.uniforms.params.getGPUBuffer(device),
      0,
      compParamsArr,
    );

    compParams2Arr[0] = canvas.width;
    compParams2Arr[1] = canvas.height;
    compParams2Arr[2] = 0;
    compParams2Arr[3] = 0;
    device.queue.writeBuffer(
      compositeShader.uniforms.params2.getGPUBuffer(device),
      0,
      compParams2Arr,
    );

    // Directional sun, computed EXACTLY like voxelize(): .xyz = world dir TOWARD the sun
    // (azimuth + elevation), .w = effective intensity (0 = disabled), color in uSunColor.
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    compSunArr[0] = Math.cos(a) * ce;
    compSunArr[1] = Math.sin(a) * ce;
    compSunArr[2] = Math.sin(e);
    compSunArr[3] = SunLight.enabled ? SunLight.intensity : 0;
    device.queue.writeBuffer(compositeShader.uniforms.sun.getGPUBuffer(device), 0, compSunArr);
    compSunColorArr[0] = SunLight.color[0];
    compSunColorArr[1] = SunLight.color[1];
    compSunColorArr[2] = SunLight.color[2];
    device.queue.writeBuffer(
      compositeShader.uniforms.sunColor.getGPUBuffer(device),
      0,
      compSunColorArr,
    );

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: compositeOutput.createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(compositePipeline);
    pass.setBindGroup(0, compositeGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // Change the voxel size (graininess). Destroys the old textures, rebuilds the grid.
  function setCellSize(newCellSize: number) {
    textures.voxelAlbedo.destroy();
    textures.voxelEmission.destroy();
    textures.voxelRadiance.destroy();
    buildGrid(newCellSize);
  }

  // Canvas resized: rebind the (new) G-buffer textures, recreate the canvas-sized debug +
  // cone + composite outputs, and rebuild the cone/composite bind groups.
  function recreate(
    newDepth: GPUTexture,
    newNormal: GPUTexture,
    newAlbedo: GPUTexture,
    newEmission: GPUTexture,
  ) {
    gDepth = newDepth;
    gNormal = newNormal;
    gAlbedo = newAlbedo;
    gEmission = newEmission;
    outputTexture.destroy();
    outputTexture = createOutput();
    coneOutput.destroy();
    coneOutput = createConeOutput();
    compositeOutput.destroy();
    compositeOutput = createCompositeOutput();
    buildConeGroup();
    // Rebuild the composite group: G-buffer + coneOutput changed.
    buildCompositeGroup();
  }

  return {
    params,
    coneParams,
    compositeParams,
    voxelize,
    mips,
    debug,
    cone,
    composite,
    recreate,
    setCellSize,
    get mipCount() {
      return mipCount;
    },
    get cellSize() {
      return cellSize;
    },
    get dims() {
      return { x: dimX, y: dimY, z: dimZ };
    },
    get textures() {
      return textures;
    },
    get outputTexture() {
      return outputTexture;
    },
    get coneOutputTexture() {
      return coneOutput;
    },
    get compositeOutputTexture() {
      return compositeOutput;
    },
    // Held for a future layer (Layer 4 reads albedo); exposed so the closure ref is live.
    get albedoTexture() {
      return gAlbedo;
    },
  };
}
