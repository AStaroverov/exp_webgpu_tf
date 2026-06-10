import { hasComponent, onAdd, onSet, query } from "bitecs";
import { shaderMeta } from "./sdf.shader.ts";
import { GPUShader } from "../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../Shader";
import { projectionMatrix } from "../ResizeSystem.ts";
import { createChangeDetector } from "../ChangedDetectorSystem.ts";
import { SunLight } from "../SunLight.ts";
import { getRenderComponents, type RenderWorldLike } from "../../world.ts";

export function createDrawShapeSystem({
  device,
  world,
  shadowMapTexture,
}: {
  world: RenderWorldLike;
  device: GPUDevice;
  shadowMapTexture: GPUTexture;
}) {
  const { Blurness, Color, GlobalTransform, LightEmitter, Roundness, Shape, Translucency } =
    getRenderComponents(world);
  const gpuShader = new GPUShader(shaderMeta);

  // Set shadow map texture on shader meta before creating bind groups
  if (shadowMapTexture) {
    shaderMeta.uniforms.shadowMap.setTexture(shadowMapTexture);
  }

  // Pipeline for shadow map pass (r32float, no depth, no blending)
  // Uses autoLayout with explicit bindGroups since it doesn't use all uniforms
  const pipelineShadowMap = shadowMapTexture
    ? gpuShader.getRenderPipeline(device, "vs_shadow_map", "fs_shadow_map", {
        targetFormat: "r32float",
        autoLayout: true,
        withBlending: false,
        withDepth: false,
        bindGroups: {
          0: ["projection", "sunShadow"],
          1: ["transform", "kind", "values", "roundness", "intensity"],
        },
      })
    : null;

  // Pipeline for visual shadow effect
  const pipelineShadow = gpuShader.getRenderPipeline(device, "vs_shadow", "fs_shadow", {
    withDepth: true,
  });

  // Main shape pipeline
  const pipelineSdf = gpuShader.getRenderPipeline(device, "vs_main", "fs_main", {
    withDepth: true,
  });

  // Emission pipeline: color (rgba16float, additive) + facing dir (rg16float, replace), no depth
  const pipelineEmit = gpuShader.getRenderPipeline(device, "vs_emit", "fs_emit", {
    targets: [
      { format: "rgba16float", blend: "additive" },
      { format: "rg16float", blend: "none" },
    ],
    autoLayout: true,
    withDepth: false,
    bindGroups: {
      0: ["projection"],
      1: [
        "transform",
        "kind",
        "color",
        "values",
        "roundness",
        "blurness",
        "intensity",
        "translucency",
      ],
    },
  });

  // Main pass bind groups (includes shadow map in group 2)
  const bindGroup0 = gpuShader.getBindGroup(device, 0);
  const bindGroup1 = gpuShader.getBindGroup(device, 1);
  const bindGroup2 = shadowMapTexture ? gpuShader.getBindGroup(device, 2) : null;

  // Shadow map pass bind groups (cached during pipeline creation)
  const shadowMapBindGroup0 = pipelineShadowMap
    ? gpuShader.getBindGroup(device, 0, "vs_shadow_map", "fs_shadow_map")
    : null;
  const shadowMapBindGroup1 = pipelineShadowMap
    ? gpuShader.getBindGroup(device, 1, "vs_shadow_map", "fs_shadow_map")
    : null;

  // Emission pass bind groups (cached during pipeline creation)
  const emitBindGroup0 = gpuShader.getBindGroup(device, 0, "vs_emit", "fs_emit");
  const emitBindGroup1 = gpuShader.getBindGroup(device, 1, "vs_emit", "fs_emit");

  // Baked z-shadows follow the shared SunLight (read each frame, no wiring):
  // xy = direction toward the sun, scaled by SQRT1_2 to keep the legacy
  // LIGHT_DIR=(-0.5,-0.5) offset magnitude; z = darkness, w = reserved.
  const SHADOW_DARKNESS = 0.4;
  const sunShadowCollect = getTypeTypedArray(shaderMeta.uniforms.sunShadow.type);

  const transformCollect = getTypeTypedArray(shaderMeta.uniforms.transform.type);
  const kindCollect = getTypeTypedArray(shaderMeta.uniforms.kind.type);
  const colorCollect = getTypeTypedArray(shaderMeta.uniforms.color.type);
  const valuesCollect = getTypeTypedArray(shaderMeta.uniforms.values.type);
  const roundnessCollect = getTypeTypedArray(shaderMeta.uniforms.roundness.type);
  const blurnessCollect = getTypeTypedArray(shaderMeta.uniforms.blurness.type);
  const intensityCollect = getTypeTypedArray(shaderMeta.uniforms.intensity.type);
  const translucencyCollect = getTypeTypedArray(shaderMeta.uniforms.translucency.type);

  const shapeChanges = createChangeDetector(world, [onAdd(Shape), onSet(Shape)]);
  const colorChanges = createChangeDetector(world, [onAdd(Color), onSet(Color)]);
  const roundnessChanges = createChangeDetector(world, [onAdd(Roundness), onSet(Roundness)]);
  const blurnessChanges = createChangeDetector(world, [onAdd(Blurness), onSet(Blurness)]);
  const translucencyChanges = createChangeDetector(world, [
    onAdd(Translucency),
    onSet(Translucency),
  ]);
  const intensityChanges = createChangeDetector(world, [onAdd(LightEmitter), onSet(LightEmitter)]);
  let prevEntityCount = 0;

  function updateBuffers() {
    const entities = query(world, [GlobalTransform, Shape, Color]); // Roundness, Shadow is optional

    if (entities.length === 0) return 0;

    const countChanged = entities.length !== prevEntityCount;
    prevEntityCount = entities.length;

    for (let i = 0; i < entities.length; i++) {
      const id = entities[i];

      transformCollect.set(GlobalTransform.matrix.getBatch(id), i * 16);

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

      if (countChanged || blurnessChanges.hasChanges()) {
        blurnessCollect[i] = hasComponent(world, id, Blurness) ? Blurness.value[id] : 0;
      }

      if (countChanged || intensityChanges.hasChanges()) {
        intensityCollect[i] = hasComponent(world, id, LightEmitter)
          ? LightEmitter.intensity[id]
          : 0;
      }

      if (countChanged || translucencyChanges.hasChanges()) {
        translucencyCollect[i] = hasComponent(world, id, Translucency) ? Translucency.value[id] : 0;
      }
    }

    sunShadowCollect[0] = Math.cos(SunLight.angle) * Math.SQRT1_2;
    sunShadowCollect[1] = -Math.sin(SunLight.angle) * Math.SQRT1_2;
    sunShadowCollect[2] = SunLight.enabled ? SHADOW_DARKNESS : 0;

    device.queue.writeBuffer(
      gpuShader.uniforms.projection.getGPUBuffer(device),
      0,
      projectionMatrix as BufferSource,
    );
    device.queue.writeBuffer(
      gpuShader.uniforms.sunShadow.getGPUBuffer(device),
      0,
      sunShadowCollect,
    );
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

    if (countChanged || blurnessChanges.hasChanges()) {
      device.queue.writeBuffer(
        gpuShader.uniforms.blurness.getGPUBuffer(device),
        0,
        blurnessCollect,
      );
    }

    if (countChanged || intensityChanges.hasChanges()) {
      device.queue.writeBuffer(
        gpuShader.uniforms.intensity.getGPUBuffer(device),
        0,
        intensityCollect,
      );
    }

    if (countChanged || translucencyChanges.hasChanges()) {
      device.queue.writeBuffer(
        gpuShader.uniforms.translucency.getGPUBuffer(device),
        0,
        translucencyCollect,
      );
    }

    return entities.length;
  }

  // Main render pass - renders shapes with shadow map sampling
  function drawShapes(renderPass: GPURenderPassEncoder) {
    const entityCount = updateBuffers();
    if (entityCount === 0) return;

    renderPass.setBindGroup(0, bindGroup0);
    renderPass.setBindGroup(1, bindGroup1);
    // Set shadow map bind group (group 2) - only needed for main shapes pass
    if (bindGroup2) {
      renderPass.setBindGroup(2, bindGroup2);
      renderPass.setPipeline(pipelineShadow);
      renderPass.draw(6, entityCount, 0, 0);
    }

    // Main shapes (samples shadow map for object-to-object shadows)
    renderPass.setPipeline(pipelineSdf);
    renderPass.draw(6, entityCount, 0, 0);

    shapeChanges.clear();
    colorChanges.clear();
    roundnessChanges.clear();
    blurnessChanges.clear();
    intensityChanges.clear();
    translucencyChanges.clear();
  }

  // Shadow map pass - renders shadow silhouettes with Z height
  function drawShadowMap(shadowMapPass: GPURenderPassEncoder) {
    const entityCount = updateBuffers();
    if (entityCount === 0 || !pipelineShadowMap || !shadowMapBindGroup0 || !shadowMapBindGroup1)
      return;

    shadowMapPass.setPipeline(pipelineShadowMap);
    shadowMapPass.setBindGroup(0, shadowMapBindGroup0);
    shadowMapPass.setBindGroup(1, shadowMapBindGroup1);
    shadowMapPass.draw(6, entityCount, 0, 0);
  }

  // Emission pass - renders emitters as premultiplied HDR color (occluders write coverage only)
  function drawEmitters(passEncoder: GPURenderPassEncoder) {
    const entityCount = updateBuffers();
    if (entityCount === 0) return;

    passEncoder.setPipeline(pipelineEmit);
    passEncoder.setBindGroup(0, emitBindGroup0);
    passEncoder.setBindGroup(1, emitBindGroup1);
    passEncoder.draw(6, entityCount, 0, 0);
  }

  return { drawShapes, drawShadowMap, drawEmitters };
}
