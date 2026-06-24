import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import {
  createRCTextures,
  WORLD_CASCADE_COUNT,
  WORLD_DIR0_W,
  WORLD_GRID_DIM,
} from "../../../WGSL/createFrame.ts";
import { shaderMeta as gatherMeta } from "./worldGather.shader.ts";
import { shaderMeta as mergeMeta } from "./worldMerge.shader.ts";
import { shaderMeta as compositeMeta } from "./worldComposite.shader.ts";
import { SunLight } from "../SunLight.ts";
import { cameraPosition, viewProjMatrix } from "../ResizeSystem.ts";
import type { SceneInstances } from "../SDFSystem/createDrawShapeSystem.ts";

type RCTextures = ReturnType<typeof createRCTextures>;

const N = WORLD_CASCADE_COUNT;

// Stage-2 world-space Radiance Cascades: a cascade HIERARCHY. Per cascade c:
// probes/side = WORLD_GRID_DIM >> c, octahedral tile side = WORLD_DIR0_W << c (so
// every cascade atlas is the same square size), cell_c = cell0 * 2^c, and the ray
// interval grows geometrically (end_c = baseInterval * 4^c). Passes:
//   (1) gather  c=0..N-1  -> probeRad[c]   (sphere-trace the SDF scene)
//   (2) merge   c=N-2..0  -> probeMerge[c] (fold coarser cascade into this one)
//   (3) composite probeMerge[0] + G-buffer -> worldLitTexture
// The far/coarse cascades resolve small distant lights (more directions, longer
// interval); merge carries that down to c0. Runs ALONGSIDE the screen-space RC.
//
// COORDINATES: world Z-up, footprints in XY, reverse-Z. Composite reconstructs
// world position with the inverse of ResizeSystem.viewProjMatrix.
export type WorldRCParams = {
  gridDim: number; // probes per side at c0 (atlas-size-defining; fixed)
  dir0W: number; // octahedral tile side at c0 (atlas-size-defining; fixed)
  cell0: number; // world units per probe at cascade 0
  probePlaneZ: number; // world height of the ground probe plane
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
// c0 (base) interval ~ one cell so cascade rings stitch without holes.
export const DEFAULT_WORLD_RC_PARAMS: WorldRCParams = {
  gridDim: WORLD_GRID_DIM,
  dir0W: WORLD_DIR0_W,
  cell0: 1.5,
  probePlaneZ: 0.5,
  gatherSteps: 48,
  intervalStart: 0,
  intervalEnd: 1.5,
  ambient: 0.2,
  sunColor: [1.0, 0.859, 0.161], // #ffdb29 warm sun
  sunIntensity: 0.1,
  sunDistance: 0.65,
  skyColor: [0.075, 0.11, 0.239], // #131c3d night sky
  skyMix: 0.32,
};

export function createWorldRadianceCascadesSystem({
  device,
  params,
  frameTextures,
  sceneTexture,
  depthTexture,
  normalTexture,
  sceneInstances,
}: {
  device: GPUDevice;
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

  // ---- per-cascade derived params ----
  const cell = (c: number) => p.cell0 * 2 ** c;
  const gridDim = (c: number) => WORLD_GRID_DIM >> c;
  const dirW = (c: number) => WORLD_DIR0_W << c;
  // interval(c): end = base * 4^c; start = (c==0)? intervalStart : end(c-1).
  const intervalEnd = (c: number) => p.intervalEnd * 4 ** c;
  const intervalStart = (c: number) => (c === 0 ? p.intervalStart : p.intervalEnd * 4 ** (c - 1));
  // Camera-snapped grid origin for cascade c (snap to its own cell so probes don't crawl).
  const originX = (c: number) => Math.floor(cameraPosition.x / cell(c)) * cell(c);
  const originY = (c: number) => Math.floor(cameraPosition.y / cell(c)) * cell(c);

  // Upload the cascade-constant gather uniforms (everything except per-frame
  // gridOrigin / instanceCount / sun).
  function uploadGatherConst(shader: GPUShader<typeof gatherMeta>, c: number) {
    writeScalar(device, shader, "gridDim", gridDim(c));
    writeScalar(device, shader, "dirW", dirW(c));
    writeScalar(device, shader, "cell", cell(c));
    writeScalar(device, shader, "probePlaneZ", p.probePlaneZ);
    writeScalar(device, shader, "intervalStart", intervalStart(c));
    writeScalar(device, shader, "intervalEnd", intervalEnd(c));
    writeScalar(device, shader, "gatherSteps", p.gatherSteps);
    writeVec4(device, shader, "sunColor", p.sunColor, p.sunIntensity, p.sunDistance);
    writeVec4(device, shader, "skyColor", p.skyColor, p.sunIntensity, p.skyMix);
  }

  // Upload the cascade-constant merge uniforms (dst = c, src = c+1).
  function uploadMergeConst(shader: GPUShader<typeof mergeMeta>, c: number) {
    writeScalar(device, shader, "gridDim", gridDim(c));
    writeScalar(device, shader, "dirW", dirW(c));
    writeScalar(device, shader, "cell", cell(c));
    writeScalar(device, shader, "gridDimSrc", gridDim(c + 1));
    writeScalar(device, shader, "dirWSrc", dirW(c + 1));
    writeScalar(device, shader, "cellSrc", cell(c + 1));
  }

  function build() {
    // === Gather: one shader per cascade (own uniforms + module). ===
    const gather = [];
    for (let c = 0; c < N; c++) {
      const shader = new GPUShader(gatherMeta);
      const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
        targetFormat: "rgba16float",
        withBlending: false,
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
      gather.push({ shader, pipeline, bindGroup0, bindGroup1, target: rcTextures.probeRad[c] });
    }

    // === Merge: one shader per cascade c=0..N-2, near=probeRad[c], far=src. ===
    // Top cascade (N-1) has no merge target; its merged value IS probeRad[N-1].
    const merge = [];
    for (let c = 0; c < N - 1; c++) {
      const src = c + 1 === N - 1 ? rcTextures.probeRad[N - 1] : rcTextures.probeMerge[c + 1];
      // Set textures BEFORE new GPUShader (the bind group snapshots them).
      mergeMeta.uniforms.nearTexture.setTexture(rcTextures.probeRad[c]);
      mergeMeta.uniforms.srcTexture.setTexture(src);
      const shader = new GPUShader(mergeMeta);
      const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
        targetFormat: "rgba16float",
        withBlending: false,
      });
      const bindGroup = shader.getBindGroup(device, 0);
      uploadMergeConst(shader, c);
      merge.push({ shader, pipeline, bindGroup, target: rcTextures.probeMerge[c] });
    }

    // === Composite: probeMerge[0] (c0, fully merged) + G-buffer -> worldLit. ===
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
    writeScalar(device, compositeShader, "probePlaneZ", p.probePlaneZ);
    writeScalar(device, compositeShader, "cell0", p.cell0);
    writeScalar(device, compositeShader, "ambient", p.ambient);

    return { gather, merge, compositeShader, compositePipeline, compositeBindGroup };
  }

  let built = build();

  function pass(
    encoder: GPUCommandEncoder,
    target: GPUTexture,
    pipeline: GPURenderPipeline,
    bindGroups: GPUBindGroup[],
  ) {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" },
      ],
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
    // (1) Gather every cascade.
    for (let c = 0; c < N; c++) {
      const g = built.gather[c];
      uploadOrigin(g.shader, "gridOrigin", c);
      writeScalar(device, g.shader, "instanceCount", sceneInstances.instanceCount);
      writeSun(device, g.shader);
      pass(encoder, g.target, g.pipeline, [g.bindGroup0, g.bindGroup1]);
    }

    // (2) Merge top-down: c = N-2 .. 0 (coarser cascade already merged).
    for (let c = N - 2; c >= 0; c--) {
      const m = built.merge[c];
      uploadOrigin(m.shader, "gridOrigin", c);
      uploadOrigin(m.shader, "gridOriginSrc", c + 1);
      pass(encoder, m.target, m.pipeline, [m.bindGroup]);
    }

    // (3) Composite from the fully merged c0.
    uploadOrigin(built.compositeShader, "gridOrigin", 0);
    writeInvViewProj(device, built.compositeShader, invViewProj);
    pass(encoder, rcTextures.worldLitTexture, built.compositePipeline, [built.compositeBindGroup]);
  }

  function destroyBuilt() {
    for (const g of built.gather) g.shader.destroy();
    for (const m of built.merge) m.shader.destroy();
    built.compositeShader.destroy();
  }

  function destroyTextures() {
    for (const t of rcTextures.probeRad) t.destroy();
    for (const t of rcTextures.probeMerge) t.destroy();
    rcTextures.worldLitTexture.destroy();
  }

  function recreate(
    canvas: HTMLCanvasElement,
    nextSceneTexture: GPUTexture,
    nextDepthTexture: GPUTexture,
    nextNormalTexture: GPUTexture,
  ) {
    destroyBuilt();
    destroyTextures();
    const next = createRCTextures(device, canvas);
    Object.assign(frameTextures, next);
    rcTextures = {
      probeRad: next.probeRad,
      probeMerge: next.probeMerge,
      worldLitTexture: next.worldLitTexture,
    };
    sceneTexture = nextSceneTexture;
    depthTexture = nextDepthTexture;
    normalTexture = nextNormalTexture;
    built = build();
  }

  function destroy() {
    destroyBuilt();
    destroyTextures();
  }

  // Live-tunable scalars. cell0/intervalEnd feed per-cascade cell/interval, so every
  // cascade's gather + merge constants are re-uploaded.
  function setParams(partial: Partial<WorldRCParams>) {
    Object.assign(p, partial);
    for (let c = 0; c < N; c++) uploadGatherConst(built.gather[c].shader, c);
    for (let c = 0; c < N - 1; c++) uploadMergeConst(built.merge[c].shader, c);
    writeScalar(device, built.compositeShader, "probePlaneZ", p.probePlaneZ);
    writeScalar(device, built.compositeShader, "cell0", p.cell0);
    writeScalar(device, built.compositeShader, "ambient", p.ambient);
  }

  return {
    run,
    recreate,
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
