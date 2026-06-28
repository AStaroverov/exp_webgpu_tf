import { hasComponent, onAdd, onSet, query } from "bitecs";
import { MAX_INSTANCE_COUNT, shaderMeta } from "./sdf.shader.ts";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { cameraRayDir, sceneLightDir, viewProjMatrix } from "../ResizeSystem.ts";
import { createChangeDetector } from "../ChangedDetectorSystem.ts";
import { getRenderComponents, type RenderWorld } from "../../world.ts";
import type { GPUVariable } from "../../../WebGPU/GPUVariable.ts";

// Live per-instance scene storage exposed by createDrawShapeSystem so the
// world-space gather pass can bind the SAME GPU buffers (no data copy). The
// fields are the GPUVariable wrappers around the instance buffers filled
// each frame by prepare(); instanceCount is the live clamped count for the
// gather loop bound (read it AFTER prepare() has run this frame).
export type SceneInstances = {
  transform: GPUVariable;
  kind: GPUVariable;
  values: GPUVariable;
  roundness: GPUVariable;
  color: GPUVariable;
  material: GPUVariable;
  // CPU-side copies of the per-instance collect arrays (the SAME references, not copies),
  // filled by prepare() each frame. The voxel scatter pass reads these post-prepare() to
  // compute per-instance world AABBs WITHOUT a GPU readback. Only the fields the AABB math
  // needs are exposed; color/material are read on the GPU by the voxel pass as before.
  cpuTransform: Float32Array;
  cpuKind: Uint32Array;
  cpuValues: Float32Array;
  cpuRoundness: Float32Array;
  readonly instanceCount: number;
};

// 2.5D true-3D-SDF draw system. Single instanced pipeline (vs_main + fs_main):
// each instance rasterizes its world-space impostor box; the fragment shader
// sphere-traces the local 3D SDF and writes real reverse-Z frag_depth.
//
// DEPTH CONVENTION — REVERSE-Z. The pipeline uses depthCompare "greater-equal"
// with depthClearValue 0 (GPUShader.ts withDepth + createFrame.ts); nearer =
// LARGER depth. viewProjMatrix (ResizeSystem.ts) already maps NEAR→1, FAR→0 and
// the shader writes frag_depth = (viewProj * hitWorld).z / .w. Keep all three in sync.
export function createDrawShapeSystem({
  device,
  world,
}: {
  world: RenderWorld;
  device: GPUDevice;
}) {
  const { Blurness, Color, GlobalTransform, LightEmitter, Roundness, Shape, Translucency } =
    getRenderComponents(world);
  const gpuShader = new GPUShader(shaderMeta);

  // Single pipeline: world-space impostor box → raymarched SDF, with depth.
  // Stage-3b G-buffer: THREE color targets — (0) albedo into renderTexture
  // (bgra8unorm), (1) world normal into normalTexture (rgba16float), (2) per-pixel
  // self-emission into emissionTexture (rgba16float). Order MUST match
  // FragmentOutput locations 0,1,2 AND createFrameTick's main-pass
  // color-attachment list (see createFrame.ts) — a mismatch is a hard WebGPU error.
  const pipelineSdf = gpuShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targets: [{ format: "bgra8unorm" }, { format: "rgba16float" }, { format: "rgba16float" }],
    withDepth: true,
  });

  // Emission pipeline: emitter/occluder map for Radiance Cascades. Two attachments:
  // (0) emission rgba16float ADDITIVE, (1) emit facing dir rg16float REPLACE. No depth.
  // autoLayout with explicit bind groups (it only reads a subset of the uniforms).
  const pipelineEmit = gpuShader.getRenderPipeline(device, "vs_emit", "fs_emit", {
    targets: [
      { format: "rgba16float", blend: "additive" },
      { format: "rg16float", blend: "none" },
    ],
    autoLayout: true,
    withDepth: false,
    bindGroups: {
      // vs_emit reads uViewProj + uRayDir; fs_emit reads uRayDir. uLightDir is
      // not referenced by the emit entry points, so it must NOT appear here — an
      // autoLayout bind group's entries must match the shader's reflected usage.
      0: ["viewProj", "rayDir"],
      1: ["transform", "kind", "values", "roundness", "color", "material"],
    },
  });

  // group 0 = uniforms (viewProj, rayDir, lightDir); group 1 = instance storage.
  const bindGroup0 = gpuShader.getBindGroup(device, 0);
  const bindGroup1 = gpuShader.getBindGroup(device, 1);

  // Emission pass bind groups (cached during emit pipeline creation).
  const emitBindGroup0 = gpuShader.getBindGroup(device, 0, "vs_emit", "fs_emit");
  const emitBindGroup1 = gpuShader.getBindGroup(device, 1, "vs_emit", "fs_emit");

  // Reverse-Z uniforms scratch (vec4: xyz used, w padding).
  const rayDirCollect = getTypeTypedArray(shaderMeta.uniforms.rayDir.type);
  const lightDirCollect = getTypeTypedArray(shaderMeta.uniforms.lightDir.type);

  const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
  const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
  const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
  const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
  const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
  // Emission material packed per instance: vec4(intensity, translucency, blurness, _).
  const materialCollect = getTypeTypedArray(shaderMeta.uniforms.material.type);

  const shapeChanges = createChangeDetector(world, [onAdd(Shape), onSet(Shape)]);
  const colorChanges = createChangeDetector(world, [onAdd(Color), onSet(Color)]);
  const roundnessChanges = createChangeDetector(world, [onAdd(Roundness), onSet(Roundness)]);
  const intensityChanges = createChangeDetector(world, [onAdd(LightEmitter), onSet(LightEmitter)]);
  const translucencyChanges = createChangeDetector(world, [
    onAdd(Translucency),
    onSet(Translucency),
  ]);
  const blurnessChanges = createChangeDetector(world, [onAdd(Blurness), onSet(Blurness)]);
  let prevEntityCount = 0;
  let preparedEntityCount = 0;
  let overflowReported = false;

  function prepare() {
    const entities = query(world, [GlobalTransform, Shape, Color]); // Roundness optional

    // The instance buffers are fixed at MAX_INSTANCE_COUNT; writing past it
    // throws "offset is out of bounds" and kills the frame. Clamp instead —
    // render what fits and warn once so the overflow is visible, not fatal.
    const count = Math.min(entities.length, MAX_INSTANCE_COUNT);
    if (entities.length > MAX_INSTANCE_COUNT && !overflowReported) {
      overflowReported = true;
      console.warn(
        `[draw-shape] ${entities.length} renderable shapes exceeds the ` +
          `${MAX_INSTANCE_COUNT} instance cap; rendering the first ${MAX_INSTANCE_COUNT}.`,
      );
    }

    preparedEntityCount = count;
    if (count === 0) return;

    const countChanged = count !== prevEntityCount;
    prevEntityCount = count;

    // Any of the three emission params changing re-packs the single material buffer.
    const materialChanged =
      intensityChanges.hasChanges() ||
      translucencyChanges.hasChanges() ||
      blurnessChanges.hasChanges();

    for (let i = 0; i < count; i++) {
      const id = entities[i];

      // Per-frame: full transform (center + yaw).
      transformCollect.set(GlobalTransform.matrix.getBatch(id), i * 16);

      if (countChanged || shapeChanges.hasChanges()) {
        kindCollect[i] = Shape.kind[id];
        valuesCollect.set(Shape.values.getBatch(id), i * 8);
      }

      if (countChanged || colorChanges.hasChanges()) {
        colorCollect.set(Color.getArray(id), i * 4);
      }

      if (countChanged || roundnessChanges.hasChanges()) {
        roundnessCollect[i] = hasComponent(world, id, Roundness) ? Roundness.value[id] : 0;
      }

      if (countChanged || materialChanged) {
        materialCollect[i * 4 + 0] = hasComponent(world, id, LightEmitter)
          ? LightEmitter.intensity[id]
          : 0;
        materialCollect[i * 4 + 1] = hasComponent(world, id, Translucency)
          ? Translucency.value[id]
          : 0;
        materialCollect[i * 4 + 2] = hasComponent(world, id, Blurness) ? Blurness.value[id] : 0;
        materialCollect[i * 4 + 3] = 0;
      }
    }

    // Per-frame uniforms: viewProj + camera ray + scene light.
    rayDirCollect[0] = cameraRayDir[0];
    rayDirCollect[1] = cameraRayDir[1];
    rayDirCollect[2] = cameraRayDir[2];
    lightDirCollect[0] = sceneLightDir[0];
    lightDirCollect[1] = sceneLightDir[1];
    lightDirCollect[2] = sceneLightDir[2];

    device.queue.writeBuffer(
      gpuShader.uniforms.viewProj.getGPUBuffer(device),
      0,
      viewProjMatrix as BufferSource,
    );
    device.queue.writeBuffer(gpuShader.uniforms.rayDir.getGPUBuffer(device), 0, rayDirCollect);
    device.queue.writeBuffer(gpuShader.uniforms.lightDir.getGPUBuffer(device), 0, lightDirCollect);
    device.queue.writeBuffer(
      gpuShader.uniforms.transform.getGPUBuffer(device),
      0,
      transformCollect,
    );

    if (countChanged || shapeChanges.hasChanges()) {
      device.queue.writeBuffer(gpuShader.uniforms.kind.getGPUBuffer(device), 0, kindCollect);
      device.queue.writeBuffer(gpuShader.uniforms.values.getGPUBuffer(device), 0, valuesCollect);
    }

    if (countChanged || colorChanges.hasChanges()) {
      device.queue.writeBuffer(gpuShader.uniforms.color.getGPUBuffer(device), 0, colorCollect);
    }

    if (countChanged || roundnessChanges.hasChanges()) {
      device.queue.writeBuffer(
        gpuShader.uniforms.roundness.getGPUBuffer(device),
        0,
        roundnessCollect,
      );
    }

    if (countChanged || materialChanged) {
      device.queue.writeBuffer(
        gpuShader.uniforms.material.getGPUBuffer(device),
        0,
        materialCollect,
      );
    }

    shapeChanges.clear();
    colorChanges.clear();
    roundnessChanges.clear();
    intensityChanges.clear();
    translucencyChanges.clear();
    blurnessChanges.clear();
  }

  // Main render pass: one instanced draw of the impostor cube (36 verts/instance).
  function drawShapes(renderPass: GPURenderPassEncoder) {
    if (preparedEntityCount === 0) return;

    renderPass.setPipeline(pipelineSdf);
    renderPass.setBindGroup(0, bindGroup0);
    renderPass.setBindGroup(1, bindGroup1);
    renderPass.draw(36, preparedEntityCount, 0, 0);
  }

  // Emission pass: one instanced draw of the impostor cube (36 verts/instance)
  // into the RC emitter/occluder map. Same geometry as drawShapes; the fragment
  // raymarches the SDF for coverage and writes the emit convention (no depth).
  function drawEmitters(passEncoder: GPURenderPassEncoder) {
    if (preparedEntityCount === 0) return;

    passEncoder.setPipeline(pipelineEmit);
    passEncoder.setBindGroup(0, emitBindGroup0);
    passEncoder.setBindGroup(1, emitBindGroup1);
    passEncoder.draw(36, preparedEntityCount, 0, 0);
  }

  return {
    prepare,
    drawShapes,
    drawEmitters,
    // ── Stage-1 world-RC export ──────────────────────────────────────────────
    // Live per-instance storage exposed so the world-space gather pass can bind
    // the SAME GPU buffers in its own bind group (no second copy of the data).
    // These are the GPUVariable wrappers; the gather system reads
    // .getBindGroupEntry(device) / .getGPUBuffer(device) off them. Buffers
    // already carry STORAGE|COPY_DST usage — no usage changes needed.
    sceneInstances: {
      transform: gpuShader.uniforms.transform,
      kind: gpuShader.uniforms.kind,
      values: gpuShader.uniforms.values,
      roundness: gpuShader.uniforms.roundness,
      color: gpuShader.uniforms.color,
      material: gpuShader.uniforms.material,
      // CPU-side collect arrays (same references) so the voxel scatter pass can compute
      // per-instance AABBs after prepare() has filled them this frame — no GPU readback.
      cpuTransform: transformCollect as Float32Array,
      cpuKind: kindCollect as Uint32Array,
      cpuValues: valuesCollect as Float32Array,
      cpuRoundness: roundnessCollect as Float32Array,
      // Live count actually written this frame (clamped to MAX_INSTANCE_COUNT).
      // MUST be a getter — preparedEntityCount is reassigned every prepare().
      get instanceCount() {
        return preparedEntityCount;
      },
    },
  };
}
