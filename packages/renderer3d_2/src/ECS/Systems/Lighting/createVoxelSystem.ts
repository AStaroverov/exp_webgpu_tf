import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP } from "./voxelize.shader.ts";
import { shaderMeta as debugMeta } from "./voxelDebug.shader.ts";
import { shaderMeta as giMeta } from "./voxelGi.shader.ts";
import { shaderMeta as probeMeta, CASCADE_DIR_W } from "./voxelProbe.shader.ts";
import { shaderMeta as integrateMeta } from "./voxelIntegrate.shader.ts";
import { shaderMeta as resolveMeta } from "./voxelResolve.shader.ts";
import {
  createVoxelTextures,
  DEFAULT_VOXEL_GRID,
  type VoxelGridConfig,
  type VoxelTextures,
} from "./voxelResources.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";

export type VoxelParams = {
  // Ambient floor added to the debug Lambert shade.
  ambient: number;
};

export type VoxelGiParams = {
  ambient: number; // ambient floor
  numRays: number; // hemisphere rays per pixel
  maxDist: number; // secondary-ray reach (world units)
  normalBias: number; // extra lift of the secondary origin off the surface
  skyIntensity: number; // radiance returned on a secondary-ray miss
  giStrength: number; // multiplier on the gathered GI term
  accumAlpha: number; // temporal EMA blend (lower = smoother/more denoise; 1 = off)
};

// Stage 2.2 Radiance-Cascades (cascade 0) params. Probes gather at a sparse screen grid;
// the resolve composites them at full resolution.
export type VoxelRcParams = {
  ambient: number;
  maxDist: number; // probe ray reach (cascade-0 interval length)
  normalBias: number;
  skyIntensity: number;
  giStrength: number;
};

// Voxel scene system (Phase 1): voxelize() fills the 3D albedo/emission textures from the
// SDF scene each frame; debug() raymarches them into a canvas-sized output texture for
// inspection.
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
  grid = DEFAULT_VOXEL_GRID,
}: {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  sceneInstances: SceneInstances;
  // G-buffer (from the main SDF pass): used by the RC path for primary visibility.
  depthTexture: GPUTexture;
  normalTexture: GPUTexture;
  albedoTexture: GPUTexture;
  grid?: VoxelGridConfig;
}) {
  // G-buffer refs (rebound on resize via recreate()).
  let gDepth = depthTexture;
  let gNormal = normalTexture;
  let gAlbedo = albedoTexture;
  const params: VoxelParams = { ambient: 0.12 };
  const giParams: VoxelGiParams = {
    ambient: 0.08,
    numRays: 8,
    maxDist: 24,
    normalBias: 0,
    skyIntensity: 0,
    giStrength: 1,
    accumAlpha: 0.1,
  };
  // GI renders at 1/giScale resolution then upscales on present — brute force at full
  // res is billions of textureLoads/frame and hangs the GPU. Reduce rays/res to taste.
  let giScale = 4;

  const rcParams: VoxelRcParams = {
    ambient: 0.08,
    maxDist: 24,
    normalBias: 0,
    skyIntensity: 0,
    giStrength: 1,
  };
  let probeSpacing = 8; // screen px between probes

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
  const giShader = new GPUShader(giMeta);
  // HDR accumulation target (temporal EMA needs >8-bit precision); present upscales it.
  const giPipeline = giShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });

  // RC cascade-0: probe gather (-> probeIrr) + full-res resolve (-> rcOutput).
  const probeShader = new GPUShader(probeMeta);
  const probePipeline = probeShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  const integrateShader = new GPUShader(integrateMeta);
  const integratePipeline = integrateShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  const resolveShader = new GPUShader(resolveMeta);
  const resolvePipeline = resolveShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  // Linear sampler for the bilinear probe upscale in the resolve pass.
  const probeSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
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
    ],
  });
  const voxGroup1 = device.createBindGroup({
    layout: voxPipeline.getBindGroupLayout(1),
    entries: [
      { binding: voxelizeMeta.uniforms.transform.binding, resource: { buffer: sceneInstances.transform.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.kind.binding, resource: { buffer: sceneInstances.kind.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.values.binding, resource: { buffer: sceneInstances.values.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.roundness.binding, resource: { buffer: sceneInstances.roundness.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.heights.binding, resource: { buffer: sceneInstances.heights.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.color.binding, resource: { buffer: sceneInstances.color.getGPUBuffer(device) } },
      { binding: voxelizeMeta.uniforms.material.binding, resource: { buffer: sceneInstances.material.getGPUBuffer(device) } },
    ],
  });

  // --- Scratch typed arrays for uniform uploads. ---
  const originArr = getTypeTypedArray(voxelizeMeta.uniforms.gridOrigin.type); // Float32Array(4)
  const dimsArr = getTypeTypedArray(voxelizeMeta.uniforms.gridDims.type); // Int32Array(4)
  const instanceCountArr = getTypeTypedArray(voxelizeMeta.uniforms.instanceCount.type); // Uint32Array(1)
  const paramsArr = getTypeTypedArray(debugMeta.uniforms.params.type); // Float32Array(4)
  const invViewProj = mat4.create();
  const invArr = getTypeTypedArray(debugMeta.uniforms.invViewProj.type); // Float32Array(16)
  // GI scratch.
  const giParamsArr = getTypeTypedArray(giMeta.uniforms.params.type); // Float32Array(4)
  const giParams2Arr = getTypeTypedArray(giMeta.uniforms.params2.type); // Float32Array(4)
  const giInvArr = getTypeTypedArray(giMeta.uniforms.invViewProj.type); // Float32Array(16)
  // RC scratch.
  const probeParamsArr = getTypeTypedArray(probeMeta.uniforms.params.type); // Float32Array(4)
  const probeParams2Arr = getTypeTypedArray(probeMeta.uniforms.params2.type); // Float32Array(4)
  const probeInvArr = getTypeTypedArray(probeMeta.uniforms.invViewProj.type); // Float32Array(16)
  const integrateParamsArr = getTypeTypedArray(integrateMeta.uniforms.params.type); // Float32Array(4)
  const integrateParams2Arr = getTypeTypedArray(integrateMeta.uniforms.params2.type); // Float32Array(4)
  const resolveParamsArr = getTypeTypedArray(resolveMeta.uniforms.params.type); // Float32Array(4)
  const resolveParams2Arr = getTypeTypedArray(resolveMeta.uniforms.params2.type); // Float32Array(4)
  const resolveInvArr = getTypeTypedArray(resolveMeta.uniforms.invViewProj.type); // Float32Array(16)

  // --- Grid state (rebuilt by buildGrid). ---
  let cellSize = grid.cellSize;
  let dimX = grid.dimX;
  let dimY = grid.dimY;
  let dimZ = grid.dimZ;
  let textures: VoxelTextures;
  let voxGroup2: GPUBindGroup;
  let debugGroup0: GPUBindGroup;
  let dispatchX = 0;
  let dispatchY = 0;
  let dispatchZ = 0;

  // GI temporal accumulation (ping-pong, reduced resolution, HDR). giRead[i] is the GI
  // bind group whose history binding = giAccum[i]; each frame we write giAccum[w] while
  // reading giAccum[1-w] via giRead[1-w], then swap.
  const createGiOutput = () =>
    device.createTexture({
      size: [
        Math.max(1, Math.ceil(canvas.width / giScale)),
        Math.max(1, Math.ceil(canvas.height / giScale)),
        1,
      ],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let giAccum: [GPUTexture, GPUTexture] = [createGiOutput(), createGiOutput()];
  let giRead: [GPUBindGroup, GPUBindGroup];
  let giWrite = 0; // accum index written THIS frame
  let giLast = 0; // accum index written LAST frame (the presented one)
  const prevVP = new Float32Array(16); // last frame's viewProj, for camera-move reset

  // (Re)build both GI bind groups from the current voxel textures + giAccum history.
  function buildGiReadGroups() {
    const mk = (i: number) =>
      device.createBindGroup({
        layout: giPipeline.getBindGroupLayout(0),
        entries: [
          giShader.uniforms.params.getBindGroupEntry(device),
          giShader.uniforms.params2.getBindGroupEntry(device),
          giShader.uniforms.invViewProj.getBindGroupEntry(device),
          giShader.uniforms.gridOrigin.getBindGroupEntry(device),
          giShader.uniforms.gridDims.getBindGroupEntry(device),
          { binding: giMeta.uniforms.voxelAlbedo.binding, resource: textures.voxelAlbedo.createView({ dimension: "3d" }) },
          { binding: giMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
          { binding: giMeta.uniforms.historyTex.binding, resource: giAccum[i].createView() },
        ],
      });
    giRead = [mk(0), mk(1)];
  }

  // --- RC cascade-0 textures + bind groups. probeIrr is (canvas/probeSpacing) sized;
  // rcOutput is full canvas. Both rebuilt on resize / spacing change. ---
  const probeDims = () => ({
    x: Math.max(1, Math.ceil(canvas.width / probeSpacing)),
    y: Math.max(1, Math.ceil(canvas.height / probeSpacing)),
  });
  const createProbeTex = () => {
    const d = probeDims();
    return device.createTexture({
      size: [d.x, d.y, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  };
  const createRcOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  // Directional probe atlas: (probesX*DIR_W) × (probesY*DIR_W), one texel per (probe,dir).
  const createCascadeAtlas = () => {
    const d = probeDims();
    return device.createTexture({
      size: [d.x * CASCADE_DIR_W, d.y * CASCADE_DIR_W, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  };
  let probeIrr = createProbeTex();
  let rcOutput = createRcOutput();
  let cascadeAtlas = createCascadeAtlas();
  let probeGroup0: GPUBindGroup;
  let integrateGroup0: GPUBindGroup;
  let resolveGroup0: GPUBindGroup;

  // (Re)build the probe + resolve bind groups from the current voxel textures + probeIrr.
  function buildRcGroups() {
    probeGroup0 = device.createBindGroup({
      layout: probePipeline.getBindGroupLayout(0),
      entries: [
        probeShader.uniforms.params.getBindGroupEntry(device),
        probeShader.uniforms.params2.getBindGroupEntry(device),
        probeShader.uniforms.invViewProj.getBindGroupEntry(device),
        probeShader.uniforms.gridOrigin.getBindGroupEntry(device),
        probeShader.uniforms.gridDims.getBindGroupEntry(device),
        { binding: probeMeta.uniforms.voxelAlbedo.binding, resource: textures.voxelAlbedo.createView({ dimension: "3d" }) },
        { binding: probeMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
        { binding: probeMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: probeMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
      ],
    });
    integrateGroup0 = device.createBindGroup({
      layout: integratePipeline.getBindGroupLayout(0),
      entries: [
        integrateShader.uniforms.params.getBindGroupEntry(device),
        integrateShader.uniforms.params2.getBindGroupEntry(device),
        { binding: integrateMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: integrateMeta.uniforms.cascadeAtlas.binding, resource: cascadeAtlas.createView() },
      ],
    });
    resolveGroup0 = device.createBindGroup({
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [
        resolveShader.uniforms.params.getBindGroupEntry(device),
        resolveShader.uniforms.params2.getBindGroupEntry(device),
        resolveShader.uniforms.invViewProj.getBindGroupEntry(device),
        resolveShader.uniforms.gridOrigin.getBindGroupEntry(device),
        resolveShader.uniforms.gridDims.getBindGroupEntry(device),
        { binding: resolveMeta.uniforms.albedoTex.binding, resource: gAlbedo.createView() },
        { binding: resolveMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: resolveMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
        { binding: resolveMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
        { binding: resolveMeta.uniforms.probeIrr.binding, resource: probeIrr.createView() },
        { binding: resolveMeta.uniforms.probeSampler.binding, resource: probeSampler },
      ],
    });
  }

  // (Re)build the voxel textures + the two texture-referencing bind groups for the
  // current cellSize, and upload the grid uniforms to both shaders.
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
    voxGroup2 = device.createBindGroup({
      layout: voxPipeline.getBindGroupLayout(2),
      entries: [
        { binding: voxelizeMeta.uniforms.voxelAlbedo.binding, resource: textures.voxelAlbedo.createView({ dimension: "3d" }) },
        { binding: voxelizeMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
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
        { binding: debugMeta.uniforms.voxelAlbedo.binding, resource: textures.voxelAlbedo.createView({ dimension: "3d" }) },
        { binding: debugMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
      ],
    });

    // GI bind groups reference the same voxel textures + the (already-created) giAccum.
    buildGiReadGroups();
    // RC bind groups reference the same voxel textures + the (already-created) probeIrr.
    buildRcGroups();

    // Grid uniforms (shared by both shaders).
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
    device.queue.writeBuffer(giShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(giShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    device.queue.writeBuffer(probeShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(probeShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    device.queue.writeBuffer(resolveShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(resolveShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);

    dispatchX = Math.ceil(dimX / WORKGROUP);
    dispatchY = Math.ceil(dimY / WORKGROUP);
    dispatchZ = Math.ceil(dimZ / WORKGROUP);
  }
  buildGrid(cellSize);

  // Canvas-sized debug output (presented). Independent of the voxel grid.
  const createOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "bgra8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let outputTexture = createOutput();

  // Re-voxelize the scene into the 3D textures (run before debug()/the GI gather).
  function voxelize(encoder: GPUCommandEncoder) {
    instanceCountArr[0] = sceneInstances.instanceCount;
    device.queue.writeBuffer(
      voxShader.uniforms.instanceCount.getGPUBuffer(device),
      0,
      instanceCountArr,
    );

    const pass = encoder.beginComputePass();
    pass.setPipeline(voxPipeline);
    pass.setBindGroup(0, voxGroup0);
    pass.setBindGroup(1, voxGroup1);
    pass.setBindGroup(2, voxGroup2);
    pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    pass.end();
  }

  // Raymarch the voxel grid into outputTexture.
  function debug(encoder: GPUCommandEncoder) {
    paramsArr[0] = params.ambient;
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

  // Brute-force voxel GI (Stage 2.1a + 2.1b temporal). Ping-pong: write giAccum[giWrite]
  // while reading giAccum[1-giWrite] as history. frameIndex feeds the per-pixel jitter so
  // successive frames average different rays. The EMA is reset (alpha=1) the frame the
  // camera moved, since the co-located history is then stale (no reprojection yet).
  function gi(encoder: GPUCommandEncoder, frameIndex: number) {
    mat4.invert(invViewProj, viewProjMatrix);
    giInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(giShader.uniforms.invViewProj.getGPUBuffer(device), 0, giInvArr);

    // Camera-move detection against the previous frame's viewProj.
    let moved = false;
    for (let i = 0; i < 16; i++) {
      if (prevVP[i] !== viewProjMatrix[i]) {
        moved = true;
        break;
      }
    }
    prevVP.set(viewProjMatrix as Float32Array);

    giParamsArr[0] = giParams.ambient;
    giParamsArr[1] = giParams.numRays;
    giParamsArr[2] = giParams.maxDist;
    giParamsArr[3] = giParams.normalBias;
    device.queue.writeBuffer(giShader.uniforms.params.getGPUBuffer(device), 0, giParamsArr);

    giParams2Arr[0] = frameIndex;
    giParams2Arr[1] = giParams.skyIntensity;
    giParams2Arr[2] = giParams.giStrength;
    giParams2Arr[3] = moved ? 1 : giParams.accumAlpha;
    device.queue.writeBuffer(giShader.uniforms.params2.getGPUBuffer(device), 0, giParams2Arr);

    const write = giWrite;
    const read = 1 - write;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: giAccum[write].createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(giPipeline);
    pass.setBindGroup(0, giRead[read]); // history = giAccum[read]
    pass.draw(6, 1, 0, 0);
    pass.end();

    giLast = write;
    giWrite = read;
  }

  // RC (Stage 2.3a — directional, single cascade): probe gather -> directional atlas,
  // integrate (cosine-weighted) -> probeIrr, full-res resolve -> rcOutput.
  function rc(encoder: GPUCommandEncoder) {
    mat4.invert(invViewProj, viewProjMatrix);

    // Probe pass uniforms: params = (spacing, maxDist, normalBias, sky).
    probeParamsArr[0] = probeSpacing;
    probeParamsArr[1] = rcParams.maxDist;
    probeParamsArr[2] = rcParams.normalBias;
    probeParamsArr[3] = rcParams.skyIntensity;
    device.queue.writeBuffer(probeShader.uniforms.params.getGPUBuffer(device), 0, probeParamsArr);
    probeParams2Arr[0] = canvas.width;
    probeParams2Arr[1] = canvas.height;
    probeParams2Arr[2] = 0; // interval start (cascade 0)
    device.queue.writeBuffer(probeShader.uniforms.params2.getGPUBuffer(device), 0, probeParams2Arr);
    probeInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(probeShader.uniforms.invViewProj.getGPUBuffer(device), 0, probeInvArr);

    // Integrate pass uniforms.
    integrateParamsArr[0] = probeSpacing;
    device.queue.writeBuffer(integrateShader.uniforms.params.getGPUBuffer(device), 0, integrateParamsArr);
    integrateParams2Arr[0] = canvas.width;
    integrateParams2Arr[1] = canvas.height;
    device.queue.writeBuffer(integrateShader.uniforms.params2.getGPUBuffer(device), 0, integrateParams2Arr);

    // Resolve pass uniforms.
    resolveParamsArr[0] = rcParams.ambient;
    resolveParamsArr[1] = rcParams.giStrength;
    device.queue.writeBuffer(resolveShader.uniforms.params.getGPUBuffer(device), 0, resolveParamsArr);
    resolveParams2Arr[0] = probeSpacing;
    device.queue.writeBuffer(resolveShader.uniforms.params2.getGPUBuffer(device), 0, resolveParams2Arr);
    resolveInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(resolveShader.uniforms.invViewProj.getGPUBuffer(device), 0, resolveInvArr);

    // 1. Probe gather -> directional atlas.
    const probePass = encoder.beginRenderPass({
      colorAttachments: [
        { view: cascadeAtlas.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
      ],
    });
    probePass.setPipeline(probePipeline);
    probePass.setBindGroup(0, probeGroup0);
    probePass.draw(6, 1, 0, 0);
    probePass.end();

    // 2. Integrate atlas tiles -> probeIrr.
    const integratePass = encoder.beginRenderPass({
      colorAttachments: [
        { view: probeIrr.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
      ],
    });
    integratePass.setPipeline(integratePipeline);
    integratePass.setBindGroup(0, integrateGroup0);
    integratePass.draw(6, 1, 0, 0);
    integratePass.end();

    // 3. Full-res resolve -> rcOutput.
    const resolvePass = encoder.beginRenderPass({
      colorAttachments: [
        { view: rcOutput.createView(), clearValue: [0, 0, 0, 1], loadOp: "clear", storeOp: "store" },
      ],
    });
    resolvePass.setPipeline(resolvePipeline);
    resolvePass.setBindGroup(0, resolveGroup0);
    resolvePass.draw(6, 1, 0, 0);
    resolvePass.end();
  }

  // Change the voxel size (graininess). Destroys the old textures, rebuilds the grid.
  function setCellSize(newCellSize: number) {
    textures.voxelAlbedo.destroy();
    textures.voxelEmission.destroy();
    buildGrid(newCellSize);
  }

  // Rebuild the ping-pong GI accumulation textures + their bind groups.
  function rebuildGiAccum() {
    giAccum[0].destroy();
    giAccum[1].destroy();
    giAccum = [createGiOutput(), createGiOutput()];
    giWrite = 0;
    giLast = 0;
    buildGiReadGroups();
  }

  // Canvas resized: rebind the (recreated) G-buffer + recreate the debug output, GI accum,
  // and RC probe/output textures.
  function recreate(newDepth: GPUTexture, newNormal: GPUTexture, newAlbedo: GPUTexture) {
    gDepth = newDepth;
    gNormal = newNormal;
    gAlbedo = newAlbedo;
    outputTexture.destroy();
    outputTexture = createOutput();
    rebuildGiAccum();
    probeIrr.destroy();
    probeIrr = createProbeTex();
    rcOutput.destroy();
    rcOutput = createRcOutput();
    cascadeAtlas.destroy();
    cascadeAtlas = createCascadeAtlas();
    buildRcGroups();
  }

  // Change the probe spacing (px between probes). Rebuilds probeIrr + atlas + RC groups.
  function setProbeSpacing(spacing: number) {
    probeSpacing = Math.max(1, Math.round(spacing));
    probeIrr.destroy();
    probeIrr = createProbeTex();
    cascadeAtlas.destroy();
    cascadeAtlas = createCascadeAtlas();
    buildRcGroups();
  }

  // Change the GI resolution divisor (1 = full res, 4 = quarter, …). Bigger = cheaper.
  function setGiScale(scale: number) {
    giScale = Math.max(1, Math.round(scale));
    rebuildGiAccum();
  }

  return {
    params,
    giParams,
    rcParams,
    voxelize,
    debug,
    gi,
    rc,
    recreate,
    setCellSize,
    setGiScale,
    setProbeSpacing,
    get giScale() {
      return giScale;
    },
    get probeSpacing() {
      return probeSpacing;
    },
    get rcOutputTexture() {
      return rcOutput;
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
    get giOutputTexture() {
      return giAccum[giLast];
    },
  };
}
