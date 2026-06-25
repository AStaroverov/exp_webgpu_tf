import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { viewProjMatrix } from "../ResizeSystem.ts";
import { shaderMeta as voxelizeMeta, WORKGROUP } from "./voxelize.shader.ts";
import { shaderMeta as debugMeta } from "./voxelDebug.shader.ts";
import { shaderMeta as giMeta } from "./voxelGi.shader.ts";
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

// Voxel scene system (Phase 1): voxelize() fills the 3D albedo/emission textures from the
// SDF scene each frame; debug() raymarches them into a canvas-sized output texture for
// inspection; gi() is the brute-force voxel-GI reference (the ground truth the upcoming
// Voxel Cone Tracing path — see docs/voxel-cone-tracing-impl.md — must converge to).
//
// GRANULARITY: the world box (origin + extent) is FIXED; cellSize controls voxel size
// (and thus per-axis dims = round(extent/cellSize)) — the "graininess" knob. Smaller
// cellSize = finer voxels = more of them. setCellSize() rebuilds the textures + the two
// texture-referencing bind groups; the canvas-sized output texture is independent of it.
export function createVoxelSystem({
  device,
  canvas,
  sceneInstances,
  grid = DEFAULT_VOXEL_GRID,
}: {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  sceneInstances: SceneInstances;
  grid?: VoxelGridConfig;
}) {
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
    device.queue.writeBuffer(giShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(giShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);

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

  // Canvas resized: recreate the canvas-sized debug output + the GI accumulation textures.
  function recreate() {
    outputTexture.destroy();
    outputTexture = createOutput();
    rebuildGiAccum();
  }

  // Change the GI resolution divisor (1 = full res, 4 = quarter, …). Bigger = cheaper.
  function setGiScale(scale: number) {
    giScale = Math.max(1, Math.round(scale));
    rebuildGiAccum();
  }

  return {
    params,
    giParams,
    voxelize,
    debug,
    gi,
    recreate,
    setCellSize,
    setGiScale,
    get giScale() {
      return giScale;
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
