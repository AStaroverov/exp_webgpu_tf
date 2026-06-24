import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import {
  createRCTextures,
  WORLD_CASCADE_COUNT,
  WORLD_DIR0_W,
  WORLD_GRID_X,
  WORLD_GRID_Y,
  WORLD_GRID_Z,
} from "../../../WGSL/createFrame.ts";
import { shaderMeta as gatherMeta } from "./worldGather.shader.ts";
import { shaderMeta as mergeMeta } from "./worldMerge.shader.ts";
import { shaderMeta as compositeMeta } from "./worldComposite.shader.ts";
import { SunLight } from "../SunLight.ts";
import { cameraPosition, viewProjMatrix } from "../ResizeSystem.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";

type RCTextures = ReturnType<typeof createRCTextures>;

const N = WORLD_CASCADE_COUNT;

// Stage-3 world-space Radiance Cascades: a cascade HIERARCHY × a stack of horizontal
// PROBE SHEETS (height layers, Model A). Per cascade c the xy grid halves and the
// octahedral tile doubles (atlas side stays constant per axis); per layer k a probe
// sheet sits at the FIXED world height z_k = baseZ + k*cellZ. gridZ / cellZ are
// CONSTANT across cascades — layers are fully independent through gather + merge and
// meet ONLY in the composite (trilinear-in-z over the two layers bracketing the
// receiver). Atlases are 2D-ARRAY textures (one array layer per probe sheet). Passes:
//   (1) gather  c=0..N-1, k=0..Z-1 -> probeRad[c] layer k (sphere-trace the SDF scene)
//   (2) merge   c=N-2..0, k=0..Z-1 -> probeMerge[c] layer k (fold coarser cascade in)
//   (3) composite probeMerge[0] (all Z layers) + G-buffer -> worldLitTexture
//
// UNIFORM ALIASING: a single GPUShader's uniform buffer cannot carry per-pass values
// (uLayerZ / uLayer) for many passes in one submit (queue.writeBuffer is ordered
// before the command buffer, so every pass would read the LAST write). Hence ONE
// GPUShader instance per (cascade, layer) — each owns its uniform buffers — while
// all instances of a type SHARE one compiled shader module (via getRenderPipeline's
// { shaderModule } option) to avoid re-compiling 25 gather / 20 merge modules.
//
// COORDINATES: world Z-up, footprints in XY, reverse-Z. Composite reconstructs
// world position with the inverse of ResizeSystem.viewProjMatrix.
export type WorldRCParams = {
  // ---- atlas-size-defining (changing any requires a full rebuild) ----
  gridX: number; // probes along x at cascade 0
  gridY: number; // probes along y at cascade 0
  gridZ: number; // number of horizontal probe sheets (array layers)
  dir0W: number; // octahedral tile side at c0
  // ---- live-tunable ----
  cell0: number; // world units per probe (xy) at cascade 0
  cellZ: number; // world height gap between successive probe sheets
  baseZ: number; // world height of layer 0
  gatherSteps: number; // sphere-trace step budget
  intervalStart: number; // c0 ray interval start (world units)
  intervalEnd: number; // c0 ray interval end == BASE interval; reach ~ base*4^(N-1)
  ambient: number; // omni light floor (matches overlay.shader.ts)
  // Sun/sky for the top-cascade miss term (gather). sunIntensity multiplies BOTH.
  sunColor: [number, number, number];
  sunIntensity: number;
  sunDistance: number;
  skyColor: [number, number, number];
  skyMix: number;
};

// cell0=1.5 -> reach base*4^(N-1) = 1.5*4^4 = 384 world units. intervalEnd is the
// c0 (base) interval ~ one cell so cascade rings stitch without holes. gridZ sheets
// 1.5 world units apart from baseZ=0.5 cover [0.5, 6.5] world height.
export const DEFAULT_WORLD_RC_PARAMS: WorldRCParams = {
  gridX: WORLD_GRID_X,
  gridY: WORLD_GRID_Y,
  gridZ: WORLD_GRID_Z,
  dir0W: WORLD_DIR0_W,
  cell0: 1,
  cellZ: 1,
  baseZ: 0,
  gatherSteps: 48,
  intervalStart: 0,
  intervalEnd: 0.3,
  ambient: 0.2,
  sunColor: [1.0, 0.859, 0.161], // #ffdb29 warm sun
  sunIntensity: 0.1,
  sunDistance: 0.65,
  skyColor: [0.075, 0.11, 0.239], // #131c3d night sky
  skyMix: 0.32,
};

export function createWorldRadianceCascadesSystem({
  device,
  canvas,
  params,
  frameTextures,
  sceneTexture,
  depthTexture,
  normalTexture,
  sceneInstances,
}: {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  params?: Partial<WorldRCParams>;
  frameTextures: RCTextures;
  sceneTexture: GPUTexture;
  depthTexture: GPUTexture;
  normalTexture: GPUTexture;
  sceneInstances: SceneInstances;
}) {
  const p = { ...DEFAULT_WORLD_RC_PARAMS, ...params };
  // Isolate the color arrays (spread copies array refs, GUI must not mutate defaults).
  p.sunColor = [...p.sunColor];
  p.skyColor = [...p.skyColor];

  // Stored so rebuild()/recreate() can re-make the canvas-sized textures.
  let canvasRef = canvas;

  const linearSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  let rcTextures = {
    probeRad: frameTextures.probeRad,
    probeMerge: frameTextures.probeMerge,
    worldLitTexture: frameTextures.worldLitTexture,
  };

  const invViewProj = mat4.create();

  // One compiled module per shader type, shared by every (c,k) instance so we don't
  // re-compile N*Z gather / (N-1)*Z merge modules. Built once in build().
  let gatherModule: GPUShaderModule;
  let mergeModule: GPUShaderModule;

  // The layer count `built` was actually constructed with. run()/setParams() iterate
  // THIS, not p.gridZ — lil-gui mutates p.gridZ live while dragging, but rebuild()
  // (and thus the matching instances) only fires on slider release; using p.gridZ
  // here would index past built.gather/merge and crash mid-drag.
  let builtZ = p.gridZ;

  // ---- per-cascade / per-layer derived params (Model A: z does NOT scale per cascade) ----
  const cell = (c: number) => p.cell0 * 2 ** c;
  const gx = (c: number) => Math.max(1, p.gridX >> c);
  const gy = (c: number) => Math.max(1, p.gridY >> c);
  const dirW = (c: number) => p.dir0W << c;
  // Layer k absolute world height — identical for every cascade.
  const layerZ = (k: number) => p.baseZ + k * p.cellZ;
  // interval(c): end = base * 4^c; start = (c==0)? intervalStart : end(c-1).
  const intervalEnd = (c: number) => p.intervalEnd * 4 ** c;
  const intervalStart = (c: number) => (c === 0 ? p.intervalStart : p.intervalEnd * 4 ** (c - 1));
  // Camera-snapped grid origin for cascade c (snap to its own cell so probes don't
  // crawl). XY only — z is the absolute layer height, never camera-snapped.
  const originX = (c: number) => Math.floor(cameraPosition.x / cell(c)) * cell(c);
  const originY = (c: number) => Math.floor(cameraPosition.y / cell(c)) * cell(c);

  // Upload the cascade-constant gather uniforms (everything except per-frame
  // gridOrigin / instanceCount / sun and the per-layer layerZ).
  function uploadGatherConst(shader: GPUShader<typeof gatherMeta>, c: number) {
    writeScalar(device, shader, "gridX", gx(c));
    writeScalar(device, shader, "gridY", gy(c));
    writeScalar(device, shader, "dirW", dirW(c));
    // uParams = (cell, intervalStart, intervalEnd, gatherSteps) — packed into one
    // uniform buffer to keep the gather under the 12-uniform-buffer-per-stage limit.
    writeVec4Raw(
      device,
      shader,
      "params",
      cell(c),
      intervalStart(c),
      intervalEnd(c),
      p.gatherSteps,
    );
    writeVec4(device, shader, "sunColor", p.sunColor, p.sunIntensity, p.sunDistance);
    writeVec4(device, shader, "skyColor", p.skyColor, p.sunIntensity, p.skyMix);
  }

  // Upload the cascade-constant merge uniforms (dst = c, src = c+1).
  function uploadMergeConst(shader: GPUShader<typeof mergeMeta>, c: number) {
    writeScalar(device, shader, "gridX", gx(c));
    writeScalar(device, shader, "gridY", gy(c));
    writeScalar(device, shader, "dirW", dirW(c));
    writeScalar(device, shader, "cell", cell(c));
    writeScalar(device, shader, "gridXSrc", gx(c + 1));
    writeScalar(device, shader, "gridYSrc", gy(c + 1));
    writeScalar(device, shader, "dirWSrc", dirW(c + 1));
    writeScalar(device, shader, "cellSrc", cell(c + 1));
  }

  // Render target view for array layer k of a 2D-array atlas (pitfall 2).
  function layerView(texture: GPUTexture, k: number): GPUTextureView {
    return texture.createView({ dimension: "2d", baseArrayLayer: k, arrayLayerCount: 1 });
  }

  type GatherInstance = {
    shader: GPUShader<typeof gatherMeta>;
    pipeline: GPURenderPipeline;
    bindGroup0: GPUBindGroup;
    bindGroup1: GPUBindGroup;
    view: GPUTextureView;
  };
  type MergeInstance = {
    shader: GPUShader<typeof mergeMeta>;
    pipeline: GPURenderPipeline;
    bindGroup: GPUBindGroup;
    view: GPUTextureView;
  };

  function build() {
    const Z = p.gridZ;
    builtZ = Z;

    // === Gather: one GPUShader per (cascade c, layer k), all sharing one module. ===
    // gather[c][k] renders into probeRad[c] array layer k; each carries its own
    // uLayerZ. Group 1 (per-instance scene storage) is rebuilt per instance.
    gatherModule = new GPUShader(gatherMeta).getShaderModule(device);
    const gather: GatherInstance[][] = [];
    for (let c = 0; c < N; c++) {
      const perLayer: GatherInstance[] = [];
      for (let k = 0; k < Z; k++) {
        const shader = new GPUShader(gatherMeta);
        const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
          targetFormat: "rgba16float",
          withBlending: false,
          shaderModule: gatherModule,
        });
        const bindGroup0 = shader.getBindGroup(device, 0);
        // Group 1 = the draw system's per-instance buffers (same declaration order).
        const bindGroup1 = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(1),
          entries: [
            sceneInstances.transform.getBindGroupEntry(device),
            sceneInstances.kind.getBindGroupEntry(device),
            sceneInstances.values.getBindGroupEntry(device),
            sceneInstances.roundness.getBindGroupEntry(device),
            sceneInstances.heights.getBindGroupEntry(device),
            sceneInstances.color.getBindGroupEntry(device),
            sceneInstances.material.getBindGroupEntry(device),
          ],
        });
        uploadGatherConst(shader, c);
        writeScalar(device, shader, "layerZ", layerZ(k));
        perLayer.push({
          shader,
          pipeline,
          bindGroup0,
          bindGroup1,
          view: layerView(rcTextures.probeRad[c], k),
        });
      }
      gather.push(perLayer);
    }

    // === Merge: one GPUShader per (cascade c=0..N-2, layer k), sharing one module. ===
    // near=probeRad[c], far=src (probeMerge[c+1] or probeRad[N-1] for the top). The
    // textures are the WHOLE 2D-array (bound via the 2d-array default view); the layer
    // is selected per pass by uLayer. Top cascade (N-1) has no merge target.
    mergeMeta.uniforms.nearTexture.setTexture(rcTextures.probeRad[0]);
    mergeMeta.uniforms.srcTexture.setTexture(rcTextures.probeRad[N - 1]);
    mergeModule = new GPUShader(mergeMeta).getShaderModule(device);
    const merge: MergeInstance[][] = [];
    for (let c = 0; c < N - 1; c++) {
      const src = c + 1 === N - 1 ? rcTextures.probeRad[N - 1] : rcTextures.probeMerge[c + 1];
      const perLayer: MergeInstance[] = [];
      for (let k = 0; k < Z; k++) {
        // Set textures BEFORE new GPUShader (the bind group snapshots them); all
        // layers of one (c) share the same array textures.
        mergeMeta.uniforms.nearTexture.setTexture(rcTextures.probeRad[c]);
        mergeMeta.uniforms.srcTexture.setTexture(src);
        const shader = new GPUShader(mergeMeta);
        const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
          targetFormat: "rgba16float",
          withBlending: false,
          shaderModule: mergeModule,
        });
        const bindGroup = shader.getBindGroup(device, 0);
        uploadMergeConst(shader, c);
        writeScalar(device, shader, "layer", k);
        perLayer.push({
          shader,
          pipeline,
          bindGroup,
          view: layerView(rcTextures.probeMerge[c], k),
        });
      }
      merge.push(perLayer);
    }

    // === Composite: probeMerge[0] (all Z layers, fully merged) + G-buffer -> worldLit. ===
    compositeMeta.uniforms.inputSampler.setSampler(linearSampler);
    compositeMeta.uniforms.sceneTexture.setTexture(sceneTexture);
    compositeMeta.uniforms.normalTexture.setTexture(normalTexture);
    compositeMeta.uniforms.depthTexture.setTexture(depthTexture);
    compositeMeta.uniforms.probeMerge.setTexture(rcTextures.probeMerge[0]);
    const compositeShader = new GPUShader(compositeMeta);
    const compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "bgra8unorm",
      withBlending: false,
    });
    const compositeBindGroup = compositeShader.getBindGroup(device, 0);
    writeScalar(device, compositeShader, "gridX", p.gridX);
    writeScalar(device, compositeShader, "gridY", p.gridY);
    writeScalar(device, compositeShader, "gridZ", p.gridZ);
    writeScalar(device, compositeShader, "baseZ", p.baseZ);
    writeScalar(device, compositeShader, "cellZ", p.cellZ);
    writeScalar(device, compositeShader, "cell0", p.cell0);
    writeScalar(device, compositeShader, "ambient", p.ambient);

    return { gather, merge, compositeShader, compositePipeline, compositeBindGroup };
  }

  let built = build();

  function passView(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    pipeline: GPURenderPipeline,
    bindGroups: GPUBindGroup[],
  ) {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{ view, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
    });
    renderPass.setPipeline(pipeline);
    for (let g = 0; g < bindGroups.length; g++) renderPass.setBindGroup(g, bindGroups[g]);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();
  }

  // Upload a camera-snapped grid origin (cascade c) into `key` of `shader`.
  function uploadOrigin(shader: GPUShader<any>, key: string, c: number) {
    const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
    buffer[0] = originX(c);
    buffer[1] = originY(c);
    buffer[2] = 0;
    buffer[3] = 0;
    device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
  }

  function run(encoder: GPUCommandEncoder, _delta: number) {
    const Z = builtZ; // NOT p.gridZ — see builtZ note (live drag vs built instances).
    const instanceCount = sceneInstances.instanceCount;

    // (1) Gather every cascade × layer. Origin/instanceCount/sun are layer-independent
    // but live in each instance's own uniform buffer, so write them per (c,k).
    for (let c = 0; c < N; c++) {
      for (let k = 0; k < Z; k++) {
        const g = built.gather[c][k];
        uploadOrigin(g.shader, "gridOrigin", c);
        writeScalar(device, g.shader, "instanceCount", instanceCount);
        writeSun(device, g.shader);
        passView(encoder, g.view, g.pipeline, [g.bindGroup0, g.bindGroup1]);
      }
    }

    // (2) Merge top-down: c = N-2 .. 0 (coarser cascade already merged), each layer.
    for (let c = N - 2; c >= 0; c--) {
      for (let k = 0; k < Z; k++) {
        const m = built.merge[c][k];
        uploadOrigin(m.shader, "gridOrigin", c);
        uploadOrigin(m.shader, "gridOriginSrc", c + 1);
        passView(encoder, m.view, m.pipeline, [m.bindGroup]);
      }
    }

    // (3) Composite from the fully merged c0 (all Z layers, trilinear-in-z).
    uploadOrigin(built.compositeShader, "gridOrigin", 0);
    writeInvViewProj(device, built.compositeShader, invViewProj);
    passView(encoder, rcTextures.worldLitTexture.createView(), built.compositePipeline, [
      built.compositeBindGroup,
    ]);
  }

  function destroyBuilt() {
    for (const perLayer of built.gather) for (const g of perLayer) g.shader.destroy();
    for (const perLayer of built.merge) for (const m of perLayer) m.shader.destroy();
    built.compositeShader.destroy();
  }

  function destroyTextures() {
    for (const t of rcTextures.probeRad) t.destroy();
    for (const t of rcTextures.probeMerge) t.destroy();
    rcTextures.worldLitTexture.destroy();
  }

  function adoptTextures(next: RCTextures) {
    Object.assign(frameTextures, next);
    rcTextures = {
      probeRad: next.probeRad,
      probeMerge: next.probeMerge,
      worldLitTexture: next.worldLitTexture,
    };
  }

  function recreate(
    canvas: HTMLCanvasElement,
    nextSceneTexture: GPUTexture,
    nextDepthTexture: GPUTexture,
    nextNormalTexture: GPUTexture,
  ) {
    canvasRef = canvas;
    destroyBuilt();
    destroyTextures();
    adoptTextures(createRCTextures(device, canvas, dims()));
    sceneTexture = nextSceneTexture;
    depthTexture = nextDepthTexture;
    normalTexture = nextNormalTexture;
    built = build();
  }

  // Full rebuild for atlas-size-defining params (gridX/gridY/gridZ): atlas dimensions
  // and the per-(c,k) instance counts change, so textures + shaders are re-made.
  function rebuild(partial: Partial<Pick<WorldRCParams, "gridX" | "gridY" | "gridZ">>) {
    Object.assign(p, partial);
    destroyBuilt();
    destroyTextures();
    adoptTextures(createRCTextures(device, canvasRef, dims()));
    built = build();
  }

  function dims() {
    return { gridX: p.gridX, gridY: p.gridY, gridZ: p.gridZ, dir0W: p.dir0W };
  }

  function destroy() {
    destroyBuilt();
    destroyTextures();
  }

  // Live-tunable scalars only (NOT gridX/gridY/gridZ — those need rebuild()). cell0/
  // intervalEnd feed per-cascade cell/interval, so every (c,k) gather + merge const is
  // re-uploaded; layerZ depends on baseZ/cellZ so it is re-written per layer.
  function setParams(partial: Partial<WorldRCParams>) {
    Object.assign(p, partial);
    const Z = builtZ; // iterate the BUILT layer count, not the (maybe mid-drag) p.gridZ.
    for (let c = 0; c < N; c++) {
      for (let k = 0; k < Z; k++) {
        const g = built.gather[c][k];
        uploadGatherConst(g.shader, c);
        writeScalar(device, g.shader, "layerZ", layerZ(k));
      }
    }
    for (let c = 0; c < N - 1; c++) {
      for (let k = 0; k < Z; k++) uploadMergeConst(built.merge[c][k].shader, c);
    }
    // gridX/gridY/gridZ are atlas-defining and set in build() only — writing the
    // (possibly mid-drag) p.* here could desync the composite from the built atlas.
    writeScalar(device, built.compositeShader, "baseZ", p.baseZ);
    writeScalar(device, built.compositeShader, "cellZ", p.cellZ);
    writeScalar(device, built.compositeShader, "cell0", p.cell0);
    writeScalar(device, built.compositeShader, "ambient", p.ambient);
  }

  return {
    run,
    recreate,
    rebuild,
    destroy,
    setParams,
    params: p,
    get outputTexture() {
      return rcTextures.worldLitTexture;
    },
  };
}

function writeScalar(device: GPUDevice, shader: GPUShader<any>, key: string, value: number) {
  const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
  buffer[0] = value;
  device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
}

// Write four raw scalars into a vec4 uniform (no rgb*mult shaping).
function writeVec4Raw(
  device: GPUDevice,
  shader: GPUShader<any>,
  key: string,
  x: number,
  y: number,
  z: number,
  w: number,
) {
  const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
  buffer[0] = x;
  buffer[1] = y;
  buffer[2] = z;
  buffer[3] = w;
  device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
}

function writeVec4(
  device: GPUDevice,
  shader: GPUShader<any>,
  key: string,
  rgb: [number, number, number],
  mult: number,
  w: number,
) {
  const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
  buffer[0] = rgb[0] * mult;
  buffer[1] = rgb[1] * mult;
  buffer[2] = rgb[2] * mult;
  buffer[3] = w;
  device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
}

// uSun: .x = SunLight.angle (toward-sun, screen frame), .y = enabled (0/1).
function writeSun(device: GPUDevice, shader: GPUShader<any>) {
  const buffer = getTypeTypedArray(shader.uniforms["sun"].variable.type);
  buffer[0] = SunLight.angle;
  buffer[1] = SunLight.enabled ? 1 : 0;
  buffer[2] = 0;
  buffer[3] = 0;
  device.queue.writeBuffer(shader.uniforms["sun"].getGPUBuffer(device), 0, buffer);
}

function writeInvViewProj(device: GPUDevice, shader: GPUShader<any>, scratch: mat4) {
  mat4.invert(scratch, viewProjMatrix);
  const buffer = getTypeTypedArray(shader.uniforms["invViewProj"].variable.type);
  buffer.set(scratch as Float32Array);
  device.queue.writeBuffer(shader.uniforms["invViewProj"].getGPUBuffer(device), 0, buffer);
}
