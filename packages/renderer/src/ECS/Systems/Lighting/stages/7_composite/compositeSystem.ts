import { gpuSpan } from "../../../../../gpuTimer.ts";
// VCT COMPOSITE sub-system (Layer 4 — the final lit image). Owns the composite shader/pipeline +
// its bind group, the consolidated frame UBO scratch, and the full-res HDR output texture, and
// issues the composite render pass. Extracted VERBATIM from createVoxelSystem: same WGSL, same
// UBO packing, same bind-group entries, same pass. Behavior is byte-for-byte identical.
//
// final = albedo·(ambient·AO + directSun + indirect) + emission. Reads the G-buffer
// (albedo/normal/emission/depth), the half-res cone output (indirect+AO, bilinear-upsampled), and
// the cone pass's sun cast-shadow visibility (the DF-style sun shadow). MUST run AFTER cone().
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { createCompositeShaderMeta } from "./voxelComposite.shader.ts";
import { SunLight } from "../../../SunLight.ts";
import type { VoxelBakedConfig } from "../../core/voxelConfig.ts";

// The G-buffer (the SDF draw pass output): reverse-Z depth + world normal + albedo + per-pixel
// self-emission. The composite reads albedo + emission + normal + depth.
export type GBuffer = {
  depth: GPUTexture;
  normal: GPUTexture;
  albedo: GPUTexture;
  emission: GPUTexture;
};

export type CompositeDeps = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  // Initial baked config (mutated + re-passed via rebuild()).
  config: VoxelBakedConfig;
  // Shared linear/clamp sampler (reused for the half-res cone upsample).
  voxelSampler: GPUSampler;
  // The current G-buffer set (re-fetched at group-build time so resize() picks up the new textures).
  getGBuffer: () => GBuffer;
  // The cone pass's HALF-res HDR output view (indirect+AO) — sampled by the composite.
  getConeView: () => GPUTextureView;
  // The cone pass's sun cast-shadow visibility view (the DF-style sun shadow source).
  getSunVisView: () => GPUTextureView;
  // Cone downscale factor → upsample_cone maps cone↔full res (rides params2.z each frame).
  getConeScale: () => number;
};

export function createCompositeSystem(deps: CompositeDeps) {
  const {
    device,
    canvas,
    voxelSampler,
    getGBuffer,
    getConeView,
    getSunVisView,
    getConeScale,
  } = deps;
  let config = deps.config;

  // VCT composite (Layer 4): final = albedo·(ambient·AO + directSun + indirect) + emission.
  let compositeShader = new GPUShader(createCompositeShaderMeta(config));
  let compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });
  let compositeGroup0: GPUBindGroup;

  // Composite scratch — ONE consolidated frame UBO (matches the WGSL `CompositeFrame` struct).
  // 12 f32 = 48 bytes. Field offsets (in f32 elements): params2@0, sun@4, sunColor@8. Filled
  // entirely inside composite(), one writeBuffer.
  const compFrameArr = new Float32Array(12);
  const CF_PARAMS2 = 0;
  const CF_SUN = 4;
  const CF_SUNCOLOR = 8;

  // Full-res HDR target for the Layer-4 composite (the final lit image — "final" source).
  const createCompositeOutput = () =>
    device.createTexture({
      size: [canvas.width, canvas.height, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let compositeOutput = createCompositeOutput();
  // Cached attachment view; refreshed in resize() when the texture is rebuilt on resize.
  let compositeView = compositeOutput.createView();

  // (Re)build the Layer-4 composite bind group: uniforms + the G-buffer (albedo/normal/
  // emission) + the cone output (indirect+AO). The coneSampler bilinear-upsamples the half-res
  // cone output. Rebuilt when the G-buffer / coneOutput change (resize or grid rebuild).
  function buildCompositeGroup() {
    const g = getGBuffer();
    compositeGroup0 = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        // The cone pass's sun cast-shadow visibility (the DF-style sun shadow source).
        {
          binding: compositeShader.shaderMeta.uniforms.sunVisTex.binding,
          resource: getSunVisView(),
        },
        compositeShader.uniforms.frame.getBindGroupEntry(device),
        {
          binding: compositeShader.shaderMeta.uniforms.albedoTex.binding,
          resource: g.albedo.createView(),
        },
        {
          binding: compositeShader.shaderMeta.uniforms.normalTex.binding,
          resource: g.normal.createView(),
        },
        { binding: compositeShader.shaderMeta.uniforms.coneTex.binding, resource: getConeView() },
        // Linear/clamp sampler (reuse voxelSampler) for the half-res cone upsample.
        {
          binding: compositeShader.shaderMeta.uniforms.coneSampler.binding,
          resource: voxelSampler,
        },
        {
          binding: compositeShader.shaderMeta.uniforms.emissionTex.binding,
          resource: g.emission.createView(),
        },
        // Reverse-Z camera depth (full-res dims for the cone upsample mapping).
        {
          binding: compositeShader.shaderMeta.uniforms.depthTex.binding,
          resource: g.depth.createView(),
        },
      ],
    });
  }

  // VCT composite (Layer 4): combine albedo + cone indirect/AO + direct sun + self-emission
  // into the final lit image → compositeOutput (full-res HDR). MUST run AFTER cone() (it reads
  // coneOutput + sunVis). Reads the G-buffer (albedo/normal/depth/emission).
  function composite(encoder: GPUCommandEncoder) {
    // Fill the consolidated frame UBO and upload it in ONE writeBuffer (ambient/exposure are
    // BAKED consts).
    compFrameArr[CF_PARAMS2 + 0] = canvas.width;
    compFrameArr[CF_PARAMS2 + 1] = canvas.height;
    compFrameArr[CF_PARAMS2 + 2] = getConeScale(); // cone downscale factor → upsample_cone maps cone↔full res
    compFrameArr[CF_PARAMS2 + 3] = 0;
    // Directional sun: .xyz = world dir TOWARD the sun (azimuth + elevation), .w = effective
    // intensity (0 = disabled). Same packing as voxelize's uSun.
    const a = SunLight.angle;
    const e = SunLight.elevation;
    const ce = Math.cos(e);
    compFrameArr[CF_SUN + 0] = Math.cos(a) * ce;
    compFrameArr[CF_SUN + 1] = Math.sin(a) * ce;
    compFrameArr[CF_SUN + 2] = Math.sin(e);
    compFrameArr[CF_SUN + 3] = SunLight.enabled ? SunLight.intensity : 0;
    compFrameArr[CF_SUNCOLOR + 0] = SunLight.color[0];
    compFrameArr[CF_SUNCOLOR + 1] = SunLight.color[1];
    compFrameArr[CF_SUNCOLOR + 2] = SunLight.color[2];
    device.queue.writeBuffer(compositeShader.uniforms.frame.getGPUBuffer(device), 0, compFrameArr);

    const pass = encoder.beginRenderPass({
      timestampWrites: gpuSpan("composite"),
      colorAttachments: [
        {
          view: compositeView,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(compositePipeline);
    pass.setBindGroup(0, compositeGroup0);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // Explicit, infrequent action: recompile the composite shader with the given config, recreate
  // its pipeline + bind group.
  function rebuild(newConfig: VoxelBakedConfig) {
    config = newConfig;
    compositeShader.destroy();
    compositeShader = new GPUShader(createCompositeShaderMeta(config));
    compositePipeline = compositeShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
      withBlending: false,
    });
    buildCompositeGroup();
  }

  // Rebuild the bind group only (G-buffer / coneOutput reference change without an output resize:
  // grid rebuild, cone-scale change).
  function rebindGroup() {
    buildCompositeGroup();
  }

  // Canvas resized: recreate the full-res output + view and rebuild the bind group (which samples
  // the new G-buffer + cone output).
  function resize() {
    compositeOutput.destroy();
    compositeOutput = createCompositeOutput();
    compositeView = compositeOutput.createView();
    buildCompositeGroup();
  }

  // Built last by createVoxelSystem's buildGrid()/recreate() calls to rebindGroup()/resize().

  return {
    composite,
    rebuild,
    rebindGroup,
    resize,
    getOutputView() {
      return compositeView;
    },
    getOutputTexture() {
      return compositeOutput;
    },
    get output() {
      return compositeOutput;
    },
  };
}
