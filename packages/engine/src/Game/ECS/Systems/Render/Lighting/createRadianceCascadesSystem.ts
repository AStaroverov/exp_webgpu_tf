import { GPUShader } from "renderer/src/WGSL/GPUShader.ts";
import { getTypeTypedArray } from "renderer/src/Shader/index.ts";
import { createFrameTextures, createRCTextures } from "renderer/src/WGSL/createFrame.ts";
import { shaderMeta as seedMeta } from "./seed.shader.ts";
import { shaderMeta as jfaMeta } from "./jfa.shader.ts";
import { shaderMeta as dfMeta } from "./df.shader.ts";
import { shaderMeta as rcMeta } from "./radianceCascades.shader.ts";
import { shaderMeta as overlayMeta } from "./overlay.shader.ts";
import { SunLight } from "renderer/src/ECS/Systems/SunLight.ts";

type RCTextures = ReturnType<typeof createFrameTextures>;

export type RCParams = {
  baseRayCount: number;
  rayInterval: number;
  intervalOverlap: number;
  srgb: number;
  // Sun direction/on-off live in the shared SunLight (renderer), not here —
  // the baked SDF shadows read the same source.
  sunColor: [number, number, number];
  sunIntensity: number;
  skyColor: [number, number, number];
  skyMix: number;
  sunDistance: number;
  firstCascadeIndex: number;
  emitCone: number;
  ambient: number;
  objectLightRadius: number;
  /** Global translucency (0 = opaque occluders, 1 = light passes freely);
   *  multiplies with the per-material Translucency component. */
  objectTranslucency: number;
};

// Hand-tuned via the Lighting lil-gui panel (warm directional source over cool night sky).
export const DEFAULT_RC_PARAMS: RCParams = {
  baseRayCount: 4,
  // Short intervals: compresses cascade seams toward the source, killing the
  // concentric ring artifact around small bright emitters (flashes).
  rayInterval: 0.25,
  intervalOverlap: 0.5,
  srgb: 1.95,
  sunColor: [1.0, 0.859, 0.161], // #ffdb29
  sunIntensity: 0.1,
  skyColor: [0.075, 0.11, 0.239], // #131c3d
  skyMix: 0.32,
  sunDistance: 0.65,
  firstCascadeIndex: 0,
  emitCone: 18,
  ambient: 0.1,
  // Boundary-light dilation for object pixels, in radiance texels.
  objectLightRadius: 6,
  objectTranslucency: 1,
};

export function createRadianceCascadesSystem({
  device,
  params,
  frameTextures,
  sceneTexture,
  drawEmitters,
}: {
  device: GPUDevice;
  params?: Partial<RCParams>;
  frameTextures: RCTextures;
  // What the composite multiplies the light over (default: the raw scene).
  // Pass a post-processed texture (e.g. pixelated scene) to stylize the scene
  // while keeping the light gradients smooth.
  sceneTexture?: GPUTexture;
  drawEmitters: (passEncoder: GPURenderPassEncoder) => void;
}) {
  const p = { ...DEFAULT_RC_PARAMS, ...params };

  const linearSampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  // RC-owned textures (recreated on resize). The litTexture is canvas-sized.
  let rcTextures = {
    emissionTexture: frameTextures.emissionTexture,
    emitDirTexture: frameTextures.emitDirTexture,
    seedA: frameTextures.seedA,
    seedB: frameTextures.seedB,
    dfTexture: frameTextures.dfTexture,
    cascA: frameTextures.cascA,
    cascB: frameTextures.cascB,
    litTexture: frameTextures.litTexture,
  };

  function build() {
    const rcW = rcTextures.emissionTexture.width;
    const rcH = rcTextures.emissionTexture.height;

    const jfaPasses = Math.ceil(Math.log2(Math.max(rcW, rcH))) + 1;
    const diag = Math.sqrt(rcW * rcW + rcH * rcH);
    const cascadeCount = Math.ceil(Math.log(diag) / Math.log(p.baseRayCount)) + 1;

    // === Seed pass: emissionTexture.a -> seedA ===
    seedMeta.uniforms.emissionTexture.setTexture(rcTextures.emissionTexture);
    const seedShader = new GPUShader(seedMeta);
    const seedPipeline = seedShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rg16float",
      withBlending: false,
    });
    const seedBindGroup = seedShader.getBindGroup(device, 0);

    // === JFA passes: ping-pong seedA<->seedB, offset 2^(p-1)...1 ===
    // One shader instance per pass so each has its own uOffset buffer + input texture.
    const jfaTextures = [rcTextures.seedA, rcTextures.seedB];
    const oneOverSizeBuffer = getTypeTypedArray(jfaMeta.uniforms.oneOverSize.type);
    oneOverSizeBuffer[0] = 1.0 / rcW;
    oneOverSizeBuffer[1] = 1.0 / rcH;

    const jfaStages: {
      shader: GPUShader<typeof jfaMeta>;
      pipeline: GPURenderPipeline;
      bindGroup: GPUBindGroup;
      target: GPUTexture;
    }[] = [];
    for (let i = 0; i < jfaPasses; i++) {
      const input = jfaTextures[i % 2];
      const target = jfaTextures[(i + 1) % 2];
      jfaMeta.uniforms.inputTexture.setTexture(input);

      const shader = new GPUShader(jfaMeta);
      const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
        targetFormat: "rg16float",
        withBlending: false,
      });
      const bindGroup = shader.getBindGroup(device, 0);

      const offset = Math.pow(2, jfaPasses - 1 - i);
      const uOffsetBuffer = getTypeTypedArray(jfaMeta.uniforms.uOffset.type);
      uOffsetBuffer[0] = offset;
      device.queue.writeBuffer(
        shader.uniforms.oneOverSize.getGPUBuffer(device),
        0,
        oneOverSizeBuffer,
      );
      device.queue.writeBuffer(shader.uniforms.uOffset.getGPUBuffer(device), 0, uOffsetBuffer);

      jfaStages.push({ shader, pipeline, bindGroup, target });
    }
    const jfaResult = jfaTextures[jfaPasses % 2];

    // === DF pass: jfaResult -> dfTexture ===
    dfMeta.uniforms.jfaTexture.setTexture(jfaResult);
    const dfShader = new GPUShader(dfMeta);
    const dfPipeline = dfShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "r16float",
      withBlending: false,
    });
    const dfBindGroup = dfShader.getBindGroup(device, 0);

    // === Cascade passes: cascadeIndex C-1...0, ping-pong cascA<->cascB ===
    // One shader instance per cascade so each has its own uCascadeIndex buffer + lastTexture.
    const cascTextures = [rcTextures.cascA, rcTextures.cascB];
    const resolutionBuffer = getTypeTypedArray(rcMeta.uniforms.resolution.type);
    resolutionBuffer[0] = rcW;
    resolutionBuffer[1] = rcH;

    const cascadeStages: {
      shader: GPUShader<typeof rcMeta>;
      pipeline: GPURenderPipeline;
      bindGroup: GPUBindGroup;
      target: GPUTexture;
    }[] = [];
    let writeSlot = 0;
    for (let c = cascadeCount - 1; c >= 0; c--) {
      const target = cascTextures[writeSlot];
      const last = cascTextures[(writeSlot + 1) % 2];
      rcMeta.uniforms.linearSampler.setSampler(linearSampler);
      rcMeta.uniforms.sceneTexture.setTexture(rcTextures.emissionTexture);
      rcMeta.uniforms.distanceTexture.setTexture(rcTextures.dfTexture);
      rcMeta.uniforms.lastTexture.setTexture(last);
      // Must be set BEFORE getBindGroup, else the bind-group snapshots an unset texture.
      rcMeta.uniforms.emitDirTexture.setTexture(rcTextures.emitDirTexture);
      const shader = new GPUShader(rcMeta);
      const pipeline = shader.getRenderPipeline(device, "vs_main", "fs_main", {
        targetFormat: "rgba16float",
        withBlending: false,
      });
      const bindGroup = shader.getBindGroup(device, 0);

      const cascadeIndexBuffer = getTypeTypedArray(rcMeta.uniforms.cascadeIndex.type);
      cascadeIndexBuffer[0] = c;
      device.queue.writeBuffer(
        shader.uniforms.resolution.getGPUBuffer(device),
        0,
        resolutionBuffer,
      );
      device.queue.writeBuffer(
        shader.uniforms.cascadeIndex.getGPUBuffer(device),
        0,
        cascadeIndexBuffer,
      );
      writeScalar(device, shader, "cascadeCount", cascadeCount);
      writeScalar(device, shader, "baseRayCount", p.baseRayCount);
      writeScalar(device, shader, "rayInterval", p.rayInterval);
      writeScalar(device, shader, "intervalOverlap", p.intervalOverlap);
      writeScalar(device, shader, "srgb", p.srgb);
      writeMisc(device, shader, p);
      writeScalar(device, shader, "sunAngle", SunLight.angle);
      // sunIntensity multiplies BOTH moon and sky, so intensity 0 === moonlight off
      // (mix(moon*I, sky*I, m) == I * mix(moon, sky, m)).
      writeVec4(device, shader, "sunColor", p.sunColor, p.sunIntensity, p.sunDistance);
      writeVec4(device, shader, "skyColor", p.skyColor, p.sunIntensity, p.skyMix);

      cascadeStages.push({ shader, pipeline, bindGroup, target });
      writeSlot = (writeSlot + 1) % 2;
    }
    const cascadeResult = cascadeStages[cascadeStages.length - 1].target;

    // === Composite pass: scene * (ambient + radiance) -> litTexture ===
    // Linear: radiance is rcDownscale-sized and upsampled here; scene is 1:1 (linear == nearest).
    overlayMeta.uniforms.inputSampler.setSampler(linearSampler);
    overlayMeta.uniforms.sceneTexture.setTexture(sceneTexture ?? frameTextures.renderTexture);
    overlayMeta.uniforms.radianceTexture.setTexture(cascadeResult);
    overlayMeta.uniforms.emissionTexture.setTexture(rcTextures.emissionTexture);
    const overlayShader = new GPUShader(overlayMeta);
    const overlayPipeline = overlayShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "bgra8unorm",
      withBlending: false,
    });
    const overlayBindGroup = overlayShader.getBindGroup(device, 0);
    writeScalar(device, overlayShader, "ambient", p.ambient);
    writeScalar(device, overlayShader, "objectLightRadius", p.objectLightRadius);

    return {
      seedShader,
      seedPipeline,
      seedBindGroup,
      jfaStages,
      dfShader,
      dfPipeline,
      dfBindGroup,
      cascadeStages,
      overlayShader,
      overlayPipeline,
      overlayBindGroup,
    };
  }

  let built = build();

  function pass(
    encoder: GPUCommandEncoder,
    target: GPUTexture,
    pipeline: GPURenderPipeline,
    bindGroup: GPUBindGroup,
    draw: (renderPass: GPURenderPassEncoder) => void,
  ) {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: target.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    draw(renderPass);
    renderPass.end();
  }

  function fullscreen(renderPass: GPURenderPassEncoder) {
    renderPass.draw(6, 1, 0, 0);
  }

  function run(encoder: GPUCommandEncoder, _delta: number) {
    // The shared SunLight can change any frame (GUI, day/night scripts) —
    // re-upload the sun scalars so RC and the baked SDF shadows never diverge.
    for (const stage of built.cascadeStages) {
      writeScalar(device, stage.shader, "sunAngle", SunLight.angle);
      writeMisc(device, stage.shader, p);
    }

    // Emit pass: emitters (additive) -> emissionTexture, facing dir (replace) -> emitDirTexture
    const emitPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: rcTextures.emissionTexture.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: rcTextures.emitDirTexture.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    drawEmitters(emitPass);
    emitPass.end();

    // Seed
    pass(encoder, rcTextures.seedA, built.seedPipeline, built.seedBindGroup, fullscreen);

    // JFA ping-pong
    for (const stage of built.jfaStages) {
      pass(encoder, stage.target, stage.pipeline, stage.bindGroup, fullscreen);
    }

    // Distance field
    pass(encoder, rcTextures.dfTexture, built.dfPipeline, built.dfBindGroup, fullscreen);

    // Cascades C-1...0
    for (const stage of built.cascadeStages) {
      pass(encoder, stage.target, stage.pipeline, stage.bindGroup, fullscreen);
    }

    // Composite -> litTexture
    pass(encoder, rcTextures.litTexture, built.overlayPipeline, built.overlayBindGroup, fullscreen);
  }

  function destroyBuilt() {
    built.seedShader.destroy();
    built.dfShader.destroy();
    built.overlayShader.destroy();
    for (const stage of built.jfaStages) stage.shader.destroy();
    for (const stage of built.cascadeStages) stage.shader.destroy();
  }

  function destroyTextures() {
    rcTextures.emissionTexture.destroy();
    rcTextures.emitDirTexture.destroy();
    rcTextures.seedA.destroy();
    rcTextures.seedB.destroy();
    rcTextures.dfTexture.destroy();
    rcTextures.cascA.destroy();
    rcTextures.cascB.destroy();
    rcTextures.litTexture.destroy();
  }

  function recreate(canvas: HTMLCanvasElement) {
    destroyBuilt();
    destroyTextures();
    rcTextures = createRCTextures(device, canvas);
    Object.assign(frameTextures, rcTextures);
    built = build();
  }

  function destroy() {
    destroyBuilt();
    destroyTextures();
  }

  function setParams(partial: Partial<RCParams>) {
    Object.assign(p, partial);
    for (const stage of built.cascadeStages) {
      writeScalar(device, stage.shader, "rayInterval", p.rayInterval);
      writeScalar(device, stage.shader, "intervalOverlap", p.intervalOverlap);
      writeScalar(device, stage.shader, "srgb", p.srgb);
      // Whole uMisc each call (incl. baked firstCascadeIndex), else live-tuning zeroes it.
      writeMisc(device, stage.shader, p);
      writeScalar(device, stage.shader, "sunAngle", SunLight.angle);
      writeVec4(device, stage.shader, "sunColor", p.sunColor, p.sunIntensity, p.sunDistance);
      writeVec4(device, stage.shader, "skyColor", p.skyColor, p.sunIntensity, p.skyMix);
    }
    writeScalar(device, built.overlayShader, "ambient", p.ambient);
    writeScalar(device, built.overlayShader, "objectLightRadius", p.objectLightRadius);
  }

  return {
    run,
    recreate,
    destroy,
    setParams,
    params: p,
    get outputTexture() {
      return rcTextures.litTexture;
    },
  };
}

function writeScalar(device: GPUDevice, shader: GPUShader<any>, key: string, value: number) {
  const buffer = getTypeTypedArray(shader.uniforms[key].variable.type);
  buffer[0] = value;
  device.queue.writeBuffer(shader.uniforms[key].getGPUBuffer(device), 0, buffer);
}

// uMisc packs four independent scalars: .x = firstCascadeIndex, .y = enableSun,
// .z = emitCone, .w = hit opacity (1 - objectTranslucency).
// (writeVec4's rgb*mult+w shape does not fit here.)
function writeMisc(device: GPUDevice, shader: GPUShader<any>, p: RCParams) {
  const buffer = getTypeTypedArray(shader.uniforms["misc"].variable.type);
  buffer[0] = p.firstCascadeIndex;
  buffer[1] = SunLight.enabled ? 1 : 0;
  buffer[2] = p.emitCone;
  buffer[3] = 1 - p.objectTranslucency;
  device.queue.writeBuffer(shader.uniforms["misc"].getGPUBuffer(device), 0, buffer);
}

// rgb * mult into .xyz, `w` packs an extra scalar (sunDistance / skyMix).
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
