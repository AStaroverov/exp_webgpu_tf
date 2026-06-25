import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP } from "./voxelize.shader.ts";
import { shaderMeta as debugMeta } from "./voxelDebug.shader.ts";
import { shaderMeta as giMeta } from "./voxelGi.shader.ts";
import { shaderMeta as probeMeta, CASCADE_DIR_W } from "./voxelProbe.shader.ts";
import { shaderMeta as integrateMeta } from "./voxelIntegrate.shader.ts";
import { shaderMeta as mergeMeta } from "./voxelMerge.shader.ts";
import { shaderMeta as blurMeta } from "./voxelBlur.shader.ts";
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

// Stage 2.3 Radiance-Cascades params. Probes gather at a sparse screen grid across a
// cascade hierarchy (each level: ×2 spacing, ×2 dirs/side, ×4 interval length); the merge
// folds far cascades into cascade 0, the resolve composites at full resolution.
export type VoxelRcParams = {
  ambient: number;
  cascadeCount: number; // number of cascades (N)
  baseInterval: number; // cascade-0 interval length (world units); len_c = base*4^c
  normalBias: number;
  skyIntensity: number;
  giStrength: number;
  weightNormal: number; // bilateral normal-similarity sharpness (merge + resolve + blur)
  weightPlane: number; // bilateral planar-depth sigma (world units)
  blurSigma: number; // edge-aware probe-grid blur radius (probes); ~0 = off
};

const MAX_CASCADES = 6;

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
    cascadeCount: 4,
    baseInterval: 2,
    normalBias: 0,
    skyIntensity: 0,
    giStrength: 1,
    weightNormal: 8,
    weightPlane: 0.5,
    blurSigma: 1.2,
  };
  let probeSpacing = 8; // screen px between cascade-0 probes
  let cascadeDir0 = CASCADE_DIR_W; // cascade-0 octahedral dirs per side (tunable)

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
  const mergeShader = new GPUShader(mergeMeta);
  const mergePipeline = mergeShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  const blurShader = new GPUShader(blurMeta);
  const blurPipeline = blurShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  const resolveShader = new GPUShader(resolveMeta);
  const resolvePipeline = resolveShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
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
  // Per-cascade / per-merge uniform contents (array<vec4<f32>,2> = 8 floats each).
  const cascadeUniformArr = new Float32Array(8);
  const mergeUniformArr = new Float32Array(8);
  const mergeParamsArr = getTypeTypedArray(mergeMeta.uniforms.params.type); // Float32Array(4)
  const blurParamsArr = getTypeTypedArray(blurMeta.uniforms.params.type); // Float32Array(4)
  const blurParams2Arr = getTypeTypedArray(blurMeta.uniforms.params2.type); // Float32Array(4)
  const blurInvArr = getTypeTypedArray(blurMeta.uniforms.invViewProj.type); // Float32Array(16)

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
  // All cascade atlases share the size (probesX0*DIR_W) × (probesY0*DIR_W): cascade c uses
  // probesX0/2^c probes × DIR_W*2^c dirs per side, keeping the product constant.
  const createAtlas = () => {
    const d = probeDims();
    return device.createTexture({
      size: [d.x * cascadeDir0, d.y * cascadeDir0, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  };
  // Per-cascade geometry. probesX_c = floor(probesX0/2^c) so probesX_c*dirsSide_c never
  // exceeds the atlas width (ceil rounding at coarse spacings would otherwise overflow).
  const cascadeInfo = (c: number) => {
    const d = probeDims();
    const f = 1 << c;
    return {
      spacing: probeSpacing * f,
      dirsSide: cascadeDir0 * f,
      probesX: Math.max(1, Math.floor(d.x / f)),
      probesY: Math.max(1, Math.floor(d.y / f)),
    };
  };

  let probeIrr = createProbeTex();
  let probeBlur = createProbeTex(); // edge-aware-blurred probe irradiance (resolve reads this)
  let rcOutput = createRcOutput();
  let rawAtlas: GPUTexture[] = []; // per cascade
  let mergedAtlas: GPUTexture[] = []; // index 0..N-2 (top needs no merge target)
  let cascadeBuf: GPUBuffer[] = []; // per-cascade probe uniform (uCascade)
  let mergeBuf: GPUBuffer[] = []; // per-merge uniform (uMerge), index 0..N-2
  let probeGroup: GPUBindGroup[] = [];
  let mergeGroup: GPUBindGroup[] = []; // index 0..N-2
  let integrateGroup0: GPUBindGroup;
  let blurGroup0: GPUBindGroup;
  let resolveGroup0: GPUBindGroup;

  // (Re)allocate cascade atlases + per-cascade uniform buffers for the current N / size.
  function allocCascades() {
    for (const t of rawAtlas) t.destroy();
    for (const t of mergedAtlas) t.destroy();
    for (const b of cascadeBuf) b.destroy();
    for (const b of mergeBuf) b.destroy();
    const N = Math.max(1, Math.min(MAX_CASCADES, Math.round(rcParams.cascadeCount)));
    rawAtlas = Array.from({ length: N }, () => createAtlas());
    mergedAtlas = Array.from({ length: Math.max(0, N - 1) }, () => createAtlas());
    cascadeBuf = Array.from({ length: N }, () =>
      device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );
    mergeBuf = Array.from({ length: Math.max(0, N - 1) }, () =>
      device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    );
  }
  allocCascades();

  // (Re)build the probe (per cascade), merge (per level), integrate and resolve bind
  // groups from the current voxel textures + G-buffer + cascade atlases.
  function buildRcGroups() {
    const N = rawAtlas.length;

    // Probe groups: shared bindings + the per-cascade uCascade buffer.
    probeGroup = rawAtlas.map((_, c) =>
      device.createBindGroup({
        layout: probePipeline.getBindGroupLayout(0),
        entries: [
          probeShader.uniforms.params.getBindGroupEntry(device),
          probeShader.uniforms.params2.getBindGroupEntry(device),
          { binding: probeMeta.uniforms.cascade.binding, resource: { buffer: cascadeBuf[c] } },
          probeShader.uniforms.invViewProj.getBindGroupEntry(device),
          probeShader.uniforms.gridOrigin.getBindGroupEntry(device),
          probeShader.uniforms.gridDims.getBindGroupEntry(device),
          { binding: probeMeta.uniforms.voxelAlbedo.binding, resource: textures.voxelAlbedo.createView({ dimension: "3d" }) },
          { binding: probeMeta.uniforms.voxelEmission.binding, resource: textures.voxelEmission.createView({ dimension: "3d" }) },
          { binding: probeMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
          { binding: probeMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        ],
      }),
    );

    // Merge groups (c = 0..N-2): raw_c + upper (merged_{c+1}, or raw_{N-1} for the top one).
    mergeGroup = mergedAtlas.map((_, c) => {
      const upper = c + 1 === N - 1 ? rawAtlas[N - 1] : mergedAtlas[c + 1];
      return device.createBindGroup({
        layout: mergePipeline.getBindGroupLayout(0),
        entries: [
          { binding: mergeMeta.uniforms.merge.binding, resource: { buffer: mergeBuf[c] } },
          mergeShader.uniforms.params.getBindGroupEntry(device),
          { binding: mergeMeta.uniforms.rawTex.binding, resource: rawAtlas[c].createView() },
          { binding: mergeMeta.uniforms.upperTex.binding, resource: upper.createView() },
          { binding: mergeMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        ],
      });
    });

    // Integrate reads cascade-0 (merged if N>1, else raw).
    const cascade0 = N > 1 ? mergedAtlas[0] : rawAtlas[0];
    integrateGroup0 = device.createBindGroup({
      layout: integratePipeline.getBindGroupLayout(0),
      entries: [
        integrateShader.uniforms.params.getBindGroupEntry(device),
        integrateShader.uniforms.params2.getBindGroupEntry(device),
        { binding: integrateMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: integrateMeta.uniforms.cascadeAtlas.binding, resource: cascade0.createView() },
      ],
    });

    // Blur reads probeIrr + G-buffer → probeBlur.
    blurGroup0 = device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [
        blurShader.uniforms.params.getBindGroupEntry(device),
        blurShader.uniforms.params2.getBindGroupEntry(device),
        blurShader.uniforms.invViewProj.getBindGroupEntry(device),
        { binding: blurMeta.uniforms.probeIn.binding, resource: probeIrr.createView() },
        { binding: blurMeta.uniforms.normalTex.binding, resource: gNormal.createView() },
        { binding: blurMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
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
        { binding: resolveMeta.uniforms.probeIrr.binding, resource: probeBlur.createView() },
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

  // RC (Stage 2.3b — cascade hierarchy + merge): N probe passes -> rawAtlas[c]; merge
  // top-down -> mergedAtlas[c]; integrate cascade-0 -> probeIrr; full-res resolve -> rcOutput.
  function rc(encoder: GPUCommandEncoder) {
    mat4.invert(invViewProj, viewProjMatrix);
    const N = rawAtlas.length;

    // Shared probe uniforms: params = (normalBias, sky), params2 = (screenW, screenH).
    probeParamsArr[0] = rcParams.normalBias;
    probeParamsArr[1] = rcParams.skyIntensity;
    device.queue.writeBuffer(probeShader.uniforms.params.getGPUBuffer(device), 0, probeParamsArr);
    probeParams2Arr[0] = canvas.width;
    probeParams2Arr[1] = canvas.height;
    device.queue.writeBuffer(probeShader.uniforms.params2.getGPUBuffer(device), 0, probeParams2Arr);
    probeInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(probeShader.uniforms.invViewProj.getGPUBuffer(device), 0, probeInvArr);

    // Per-cascade uniforms (distinct buffers → safe across passes) + interval radii.
    let r = 0;
    for (let c = 0; c < N; c++) {
      const info = cascadeInfo(c);
      const len = rcParams.baseInterval * Math.pow(4, c);
      const isTop = c === N - 1;
      cascadeUniformArr[0] = info.spacing;
      cascadeUniformArr[1] = info.dirsSide;
      cascadeUniformArr[2] = r;
      cascadeUniformArr[3] = isTop ? 1e6 : r + len;
      cascadeUniformArr[4] = isTop ? 1 : 0;
      device.queue.writeBuffer(cascadeBuf[c], 0, cascadeUniformArr);
      r += len;
    }

    // Per-merge uniforms.
    for (let c = 0; c < N - 1; c++) {
      const info = cascadeInfo(c);
      const up = cascadeInfo(c + 1);
      mergeUniformArr[0] = info.spacing;
      mergeUniformArr[1] = info.dirsSide;
      mergeUniformArr[2] = up.spacing;
      mergeUniformArr[3] = up.dirsSide;
      mergeUniformArr[4] = canvas.width;
      mergeUniformArr[5] = canvas.height;
      mergeUniformArr[6] = up.probesX;
      mergeUniformArr[7] = up.probesY;
      device.queue.writeBuffer(mergeBuf[c], 0, mergeUniformArr);
    }

    // Merge shared params (normal-similarity sharpness).
    mergeParamsArr[0] = rcParams.weightNormal;
    device.queue.writeBuffer(mergeShader.uniforms.params.getGPUBuffer(device), 0, mergeParamsArr);

    // Integrate + blur + resolve uniforms.
    integrateParamsArr[0] = probeSpacing;
    integrateParamsArr[1] = cascadeDir0;
    device.queue.writeBuffer(integrateShader.uniforms.params.getGPUBuffer(device), 0, integrateParamsArr);
    integrateParams2Arr[0] = canvas.width;
    integrateParams2Arr[1] = canvas.height;
    device.queue.writeBuffer(integrateShader.uniforms.params2.getGPUBuffer(device), 0, integrateParams2Arr);
    blurParamsArr[0] = probeSpacing;
    blurParamsArr[1] = rcParams.weightNormal;
    blurParamsArr[2] = rcParams.weightPlane;
    blurParamsArr[3] = rcParams.blurSigma;
    device.queue.writeBuffer(blurShader.uniforms.params.getGPUBuffer(device), 0, blurParamsArr);
    blurParams2Arr[0] = canvas.width;
    blurParams2Arr[1] = canvas.height;
    device.queue.writeBuffer(blurShader.uniforms.params2.getGPUBuffer(device), 0, blurParams2Arr);
    blurInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(blurShader.uniforms.invViewProj.getGPUBuffer(device), 0, blurInvArr);
    resolveParamsArr[0] = rcParams.ambient;
    resolveParamsArr[1] = rcParams.giStrength;
    resolveParamsArr[2] = rcParams.weightNormal;
    resolveParamsArr[3] = rcParams.weightPlane;
    device.queue.writeBuffer(resolveShader.uniforms.params.getGPUBuffer(device), 0, resolveParamsArr);
    resolveParams2Arr[0] = probeSpacing;
    resolveParams2Arr[1] = canvas.width;
    resolveParams2Arr[2] = canvas.height;
    device.queue.writeBuffer(resolveShader.uniforms.params2.getGPUBuffer(device), 0, resolveParams2Arr);
    resolveInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(resolveShader.uniforms.invViewProj.getGPUBuffer(device), 0, resolveInvArr);

    const fullPass = (view: GPUTextureView, pipeline: GPURenderPipeline, group: GPUBindGroup) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(6, 1, 0, 0);
      pass.end();
    };

    // 1. Probe gather per cascade -> rawAtlas[c].
    for (let c = 0; c < N; c++) {
      fullPass(rawAtlas[c].createView(), probePipeline, probeGroup[c]);
    }
    // 2. Merge top-down -> mergedAtlas[c] (reads mergedAtlas[c+1], already written).
    for (let c = N - 2; c >= 0; c--) {
      fullPass(mergedAtlas[c].createView(), mergePipeline, mergeGroup[c]);
    }
    // 3. Integrate cascade-0 tiles -> probeIrr.
    fullPass(probeIrr.createView(), integratePipeline, integrateGroup0);
    // 4. Edge-aware blur of the probe grid -> probeBlur.
    fullPass(probeBlur.createView(), blurPipeline, blurGroup0);
    // 5. Full-res resolve (bilateral upsample of probeBlur) -> rcOutput.
    fullPass(rcOutput.createView(), resolvePipeline, resolveGroup0);
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
    probeBlur.destroy();
    probeBlur = createProbeTex();
    rcOutput.destroy();
    rcOutput = createRcOutput();
    allocCascades();
    buildRcGroups();
  }

  // Change the probe spacing (cascade-0 px). Rebuilds probeIrr + cascade atlases + groups.
  function setProbeSpacing(spacing: number) {
    probeSpacing = Math.max(1, Math.round(spacing));
    probeIrr.destroy();
    probeIrr = createProbeTex();
    probeBlur.destroy();
    probeBlur = createProbeTex();
    allocCascades();
    buildRcGroups();
  }

  // Change the number of cascades. Reallocates atlases/buffers and rebinds.
  function setCascadeCount(n: number) {
    rcParams.cascadeCount = Math.max(1, Math.min(MAX_CASCADES, Math.round(n)));
    allocCascades();
    buildRcGroups();
  }

  // Change cascade-0 dirs/side (angular resolution). Resizes the atlases.
  function setCascadeDir0(n: number) {
    cascadeDir0 = Math.max(2, Math.round(n));
    allocCascades();
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
    setCascadeCount,
    setCascadeDir0,
    get cascadeDir0() {
      return cascadeDir0;
    },
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
