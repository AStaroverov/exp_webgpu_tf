import { hasComponent, onAdd, onSet, query } from "bitecs";
import { mat4 } from "gl-matrix";
import { MAX_INSTANCE_COUNT, shaderMeta } from "./sdf.shader.ts";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader";
import { cameraPosition, projectionMatrix } from "../ResizeSystem.ts";
import { createChangeDetector } from "../ChangedDetectorSystem.ts";
import { getRenderComponents, type RenderWorldLike } from "../../world.ts";

export function createDrawShapeSystem({
  device,
  world,
}: {
  world: RenderWorldLike;
  device: GPUDevice;
}) {
  const { Color, GlobalTransform, LightEmitter, Roundness, Shape } = getRenderComponents(world);
  const gpuShader = new GPUShader(shaderMeta);

  // Main G-buffer pipeline: cube-impostor SDF -> MRT (albedo + world normal +
  // emission), reverse-Z depth. Front-face cull so fragments survive when the
  // camera is inside the impostor cube.
  const pipelineSdf = gpuShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targets: [
      { format: "rgba8unorm", blend: "none" }, // @location(0) albedo
      { format: "rgba16float", blend: "none" }, // @location(1) world normal
      { format: "rgba16float", blend: "none" }, // @location(2) emission HDR
    ],
    withDepth: true,
    cullMode: "front",
    frontFace: "ccw",
  });

  const bindGroup0 = gpuShader.getBindGroup(device, 0);
  const bindGroup1 = gpuShader.getBindGroup(device, 1);

  const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
  const invTransformCollect = getTypeTypedArray(shaderMeta.uniforms.invTransform.type);
  const cameraPosCollect = new Float32Array(4); // xyz = camera world pos, w = roundSteps
  // Live-tunable render params (debug GUI). roundSteps = rounded-box sphere-trace
  // budget — the dominant per-fragment cost when zoomed in.
  const params = { roundSteps: 36 };
  const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
  const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
  const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
  const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
  const intensityCollect = getTypeTypedArray(shaderMeta.uniforms.intensity.type);

  // scratch for per-instance matrix inverse (gl-matrix writes into _inv, never
  // in place over the shared GlobalTransform view).
  const _inv = mat4.create();

  const shapeChanges = createChangeDetector(world, [onAdd(Shape), onSet(Shape)]);
  const colorChanges = createChangeDetector(world, [onAdd(Color), onSet(Color)]);
  const roundnessChanges = createChangeDetector(world, [onAdd(Roundness), onSet(Roundness)]);
  const intensityChanges = createChangeDetector(world, [onAdd(LightEmitter), onSet(LightEmitter)]);
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

    for (let i = 0; i < count; i++) {
      const id = entities[i];

      const model = GlobalTransform.matrix.getBatch(id);
      transformCollect.set(model, i * 16);

      // Inverse model (for the local-space ray + world-normal matrix). Uploaded
      // every frame because the model is. Degenerate matrices return null —
      // fall back to identity to avoid NaNs in the shader ray transform.
      if (mat4.invert(_inv, model) === null) {
        mat4.identity(_inv);
      }
      invTransformCollect.set(_inv, i * 16);

      if (countChanged || shapeChanges.hasChanges()) {
        kindCollect[i] = Shape.kind[id];
        valuesCollect.set(Shape.values.getBatch(id), i * 6);
      }

      if (countChanged || colorChanges.hasChanges()) {
        colorCollect.set(Color.getArray(id), i * 4);
      }

      if (countChanged || roundnessChanges.hasChanges()) {
        roundnessCollect[i] = Roundness.value[id];
      }

      if (countChanged || intensityChanges.hasChanges()) {
        intensityCollect[i] = hasComponent(world, id, LightEmitter)
          ? LightEmitter.intensity[id]
          : 0;
      }
    }

    cameraPosCollect[0] = cameraPosition[0];
    cameraPosCollect[1] = cameraPosition[1];
    cameraPosCollect[2] = cameraPosition[2];
    cameraPosCollect[3] = params.roundSteps;

    device.queue.writeBuffer(
      gpuShader.uniforms.projection.getGPUBuffer(device),
      0,
      projectionMatrix as BufferSource,
    );
    device.queue.writeBuffer(
      gpuShader.uniforms.cameraPos.getGPUBuffer(device),
      0,
      cameraPosCollect,
    );
    device.queue.writeBuffer(
      gpuShader.uniforms.transform.getGPUBuffer(device),
      0,
      transformCollect,
    );
    device.queue.writeBuffer(
      gpuShader.uniforms.invTransform.getGPUBuffer(device),
      0,
      invTransformCollect,
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

    if (countChanged || intensityChanges.hasChanges()) {
      device.queue.writeBuffer(
        gpuShader.uniforms.intensity.getGPUBuffer(device),
        0,
        intensityCollect,
      );
    }

    shapeChanges.clear();
    colorChanges.clear();
    roundnessChanges.clear();
    intensityChanges.clear();
  }

  // Main pass: cube-impostor SDF instances -> G-buffer. 36 verts = procedural
  // unit cube (not 6: the old quad).
  function drawShapes(renderPass: GPURenderPassEncoder) {
    if (preparedEntityCount === 0) return;

    renderPass.setBindGroup(0, bindGroup0);
    renderPass.setBindGroup(1, bindGroup1);
    renderPass.setPipeline(pipelineSdf);
    renderPass.draw(36, preparedEntityCount, 0, 0);
  }

  return { prepare, drawShapes, params };
}
