import { hasComponent, onAdd, onSet, query } from "bitecs";
import { MAX_INSTANCE_COUNT, shaderMeta } from "./sdf.shader.ts";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader/index.ts";
import { cameraRayDir, sceneLightDir, viewProjMatrix } from "../ResizeSystem.ts";
import { createChangeDetector } from "../ChangedDetectorSystem.ts";
import { getRenderComponents, type RenderWorldLike } from "../../world.ts";

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
  world: RenderWorldLike;
  device: GPUDevice;
}) {
  const { Color, GlobalTransform, Height, Roundness, Shape } = getRenderComponents(world);
  const gpuShader = new GPUShader(shaderMeta);

  // Single pipeline: world-space impostor box → raymarched SDF, with depth.
  const pipelineSdf = gpuShader.getRenderPipeline(device, "vs_main", "fs_main", {
    withDepth: true,
  });

  // group 0 = uniforms (viewProj, rayDir, lightDir); group 1 = instance storage.
  const bindGroup0 = gpuShader.getBindGroup(device, 0);
  const bindGroup1 = gpuShader.getBindGroup(device, 1);

  // Reverse-Z uniforms scratch (vec4: xyz used, w padding).
  const rayDirCollect = getTypeTypedArray(shaderMeta.uniforms.rayDir.type);
  const lightDirCollect = getTypeTypedArray(shaderMeta.uniforms.lightDir.type);

  const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
  const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
  const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
  const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
  const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
  const heightsCollect = getTypeTypedArray(shaderMeta.uniforms.heights.type);

  const shapeChanges = createChangeDetector(world, [onAdd(Shape), onSet(Shape)]);
  const colorChanges = createChangeDetector(world, [onAdd(Color), onSet(Color)]);
  const roundnessChanges = createChangeDetector(world, [onAdd(Roundness), onSet(Roundness)]);
  const heightChanges = createChangeDetector(world, [onAdd(Height), onSet(Height)]);
  let prevEntityCount = 0;
  let preparedEntityCount = 0;
  let overflowReported = false;

  function prepare() {
    const entities = query(world, [GlobalTransform, Shape, Color]); // Roundness, Height optional

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

    for (let i = 0; i < count; i++) {
      const id = entities[i];

      // Per-frame: full transform (center + yaw).
      transformCollect.set(GlobalTransform.matrix.getBatch(id), i * 16);

      if (countChanged || shapeChanges.hasChanges()) {
        kindCollect[i] = Shape.kind[id];
        valuesCollect.set(Shape.values.getBatch(id), i * 6);
      }

      if (countChanged || colorChanges.hasChanges()) {
        colorCollect.set(Color.getArray(id), i * 4);
      }

      if (countChanged || roundnessChanges.hasChanges()) {
        roundnessCollect[i] = hasComponent(world, id, Roundness) ? Roundness.value[id] : 0;
      }

      if (countChanged || heightChanges.hasChanges()) {
        heightsCollect[i] = hasComponent(world, id, Height) ? Height.value[id] : 0;
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

    if (countChanged || heightChanges.hasChanges()) {
      device.queue.writeBuffer(gpuShader.uniforms.heights.getGPUBuffer(device), 0, heightsCollect);
    }

    shapeChanges.clear();
    colorChanges.clear();
    roundnessChanges.clear();
    heightChanges.clear();
  }

  // Main render pass: one instanced draw of the impostor cube (36 verts/instance).
  function drawShapes(renderPass: GPURenderPassEncoder) {
    if (preparedEntityCount === 0) return;

    renderPass.setPipeline(pipelineSdf);
    renderPass.setBindGroup(0, bindGroup0);
    renderPass.setBindGroup(1, bindGroup1);
    renderPass.draw(36, preparedEntityCount, 0, 0);
  }

  return { prepare, drawShapes };
}
