// VCT CONE-RESOLVE sub-system (Layer 2 — the screen-probe RESOLVE + short AO cones). Owns the cone
// shader/pipeline, its two texture-referencing bind groups (coneGroup0 = uniforms + G-buffer + the
// all-mips voxelRadiance view + the screen-probe atlas; coneGroup1 = the adaptive-atlas indirection
// buffers), the per-frame uniform scratch, the shared reverse-Z inverse-VP mat4 (computed here once
// per frame — composite reuses it), and the HALF-res HDR output texture. Extracted VERBATIM from
// createVoxelSystem: same WGSL, same uniform packing, same bind-group entries, same pass, same
// dispatch. Behavior is byte-for-byte identical.
//
// The screen-probe RESOLVE (fill/bounce + emitter light via the probe SH) + short AO cones →
// coneOutput (HALF-res HDR; composite bilinear-upsamples). MUST run AFTER voxelize() + mips() +
// screenProbe(). Reads the G-buffer (depth + normal).
import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { viewProjMatrix } from "../../../ResizeSystem.ts";
import { createConeShaderMeta } from "./voxelCone.shader.ts";
import type { VoxelBakedConfig } from "../../core/voxelConfig.ts";
import type { ScreenProbeTextures } from "../../core/voxelResources.ts";
import type { createScreenProbeSystem } from "../5_screenProbe/screenProbeSystem.ts";

type ScreenProbeSystem = ReturnType<typeof createScreenProbeSystem>;

// The G-buffer subset the cone reads: reverse-Z depth + world normal (reconstruct P + N).
export type ConeGBuffer = {
  depth: GPUTexture;
  normal: GPUTexture;
};

export type ConeDeps = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  // Initial baked config (mutated + re-passed via rebuild()).
  config: VoxelBakedConfig;
  // Screen-probe sub-system — the cone resolve reads the SAME atlas/buffers the gather wrote, plus
  // its resolve params (tile / normalPow / planeK / resolveRadius) + the current parity set.
  screenProbe: ScreenProbeSystem;
  // The iso voxelRadiance texture (all mips) — bound as a 3d sampled view for the AO cones.
  getVoxelRadiance: () => GPUTexture;
  // The current G-buffer set (re-fetched at group-build time so resize() picks up the new textures).
  getGBuffer: () => ConeGBuffer;
  // Shared linear filtering sampler over the rgba16float voxelRadiance pyramid.
  voxelSampler: GPUSampler;
  // Shared grid-uniform scratch (populated by createVoxelSystem.buildGrid before rebindGrid()).
  originArr: ReturnType<typeof getTypeTypedArray>;
  dimsArr: ReturnType<typeof getTypeTypedArray>;
  // Fired whenever coneOutput is recreated (setConeScale / resize) — composite samples coneView.
  onOutputRecreated: () => void;
};

export function createConeSystem(deps: ConeDeps) {
  const {
    device,
    canvas,
    screenProbe,
    getVoxelRadiance,
    getGBuffer,
    voxelSampler,
    originArr,
    dimsArr,
    onOutputRecreated,
  } = deps;
  let config = deps.config;

  // ===== Shader + pipeline (created once; only textures/bind groups rebuild). =====
  // VCT cone GI — N-cone diffuse hemisphere gather over the G-buffer → HALF-res HDR target
  // (composite bilinear-upsamples to full res; the heavy cone work runs at ¼ the pixels).
  // Built from a factory that bakes the current config consts into the WGSL → reassignable on
  // rebuild().
  let coneShader = new GPUShader(createConeShaderMeta(config));
  let conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });

  // --- Scratch typed arrays for uniform uploads. ---
  const invViewProj = mat4.create(); // reused by cone() + composite() for inverse-viewProj
  // Cone scratch. (params/aoParams/tune are now BAKED consts — no scratch arrays.)
  const coneParams2Arr = getTypeTypedArray(coneShader.shaderMeta.uniforms.params2.type); // Float32Array(4)
  const coneInvArr = getTypeTypedArray(coneShader.shaderMeta.uniforms.invViewProj.type); // Float32Array(16)
  // Cone screen-probe-resolve lane (params3): .x = tile, .y = normalPow, .z = planeK. Uploaded in cone().
  const coneParams3Arr = getTypeTypedArray(coneShader.shaderMeta.uniforms.params3.type); // Float32Array(4)

  // Every bind group that references an atlas texture comes in TWO prebuilt parity variants
  // ([0] and [1], indexed by curSet each frame — the Stage-3 ping-pong swap is pure index flipping,
  // never a per-frame createBindGroup): the CURRENT-set consumers (cone resolve, refine, debug, the
  // gather's group-2 storage writes) bind spTex[curSet]; the gather's group-0 HISTORY bindings are
  // the one place the OTHER set (spTex[1 - curSet]) appears.
  let coneGroup0: [GPUBindGroup, GPUBindGroup];
  // The cone shader's group 1 (tileHeader + tileIndices) references the RAW probeBufs, so it is
  // rebuilt whenever probeBufs is recreated (resize / tile / adaptiveFraction change) — see
  // buildConeGroup, which reads them via screenProbe.getProbeBufs().
  let coneGroup1: GPUBindGroup;

  // (Re)build the Layer-2 cone bind group: uniforms + the G-buffer (depth/normal) + the
  // ALL-mips voxelRadiance view + the shared filtering sampler. Rebuilt whenever the
  // voxelRadiance view changes (grid rebuild) or the G-buffer changes (canvas resize).
  function buildConeGroup() {
    const { depth: gDepth, normal: gNormal } = getGBuffer();
    // The screen-probe atlas ping-pong sets + the adaptive-atlas indirection buffers are owned by the
    // screen-probe sub-system; read the current refs through its getters (the cone resolve reads the
    // SAME textures/buffers the gather wrote).
    const spTex = screenProbe.getSpTex();
    const probeBufs = screenProbe.getProbeBufs();
    // Per-parity variants (the resolve reads THIS frame's atlas = spTex[curSet]) — see the
    // ping-pong comment at the group declarations.
    const buildConeGroup0 = (tex: ScreenProbeTextures) =>
      device.createBindGroup({
        layout: conePipeline.getBindGroupLayout(0),
        entries: [
          coneShader.uniforms.params2.getBindGroupEntry(device),
          coneShader.uniforms.params3.getBindGroupEntry(device),
          coneShader.uniforms.invViewProj.getBindGroupEntry(device),
          coneShader.uniforms.gridOrigin.getBindGroupEntry(device),
          coneShader.uniforms.gridDims.getBindGroupEntry(device),
          {
            binding: coneShader.shaderMeta.uniforms.depthTex.binding,
            resource: gDepth.createView(),
          },
          {
            binding: coneShader.shaderMeta.uniforms.normalTex.binding,
            resource: gNormal.createView(),
          },
          // ALL-mips sampled view so textureSampleLevel can pick any lod. (Emitter uniforms + the 6
          // aniso volumes live in the gather's group 0 — see buildScreenProbeGroups.)
          {
            binding: coneShader.shaderMeta.uniforms.voxelRadiance.binding,
            resource: getVoxelRadiance().createView({ dimension: "3d" }),
          },
          // SCREEN-SPACE probe fill source (the diffuse fill/bounce). resolve_screen_probes point-loads
          // these (SH-L1 ×3 + the per-probe pixel/validity texture) and binds the RAW gather output
          // (shR/shG/shB) directly — the unified multi-probe smooth-kernel average supersedes the old
          // blur pass, so there is no separate blurred set. Geometry (pix) is likewise unfiltered.
          {
            binding: coneShader.shaderMeta.uniforms.screenShR.binding,
            resource: tex.shR.createView({ dimension: "2d" }),
          },
          {
            binding: coneShader.shaderMeta.uniforms.screenShG.binding,
            resource: tex.shG.createView({ dimension: "2d" }),
          },
          {
            binding: coneShader.shaderMeta.uniforms.screenShB.binding,
            resource: tex.shB.createView({ dimension: "2d" }),
          },
          {
            binding: coneShader.shaderMeta.uniforms.screenProbePix.binding,
            resource: tex.pix.createView({ dimension: "2d" }),
          },
          {
            binding: coneShader.shaderMeta.uniforms.screenProbePos.binding,
            resource: tex.pos.createView({ dimension: "2d" }),
          },
          {
            binding: coneShader.shaderMeta.uniforms.screenProbeNrm.binding,
            resource: tex.nrm.createView({ dimension: "2d" }),
          },
          { binding: coneShader.shaderMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
        ],
      });
    coneGroup0 = [buildConeGroup0(spTex[0]), buildConeGroup0(spTex[1])];
    // group 1 = the adaptive-atlas indirection (tileHeader + tileIndices, StorageRead). Bound to the
    // RAW shared buffers (not GPUVariable buffers) so the resolve sees the SAME lists the refine pass
    // wrote. Present even with no adaptive probes (tileHeader is all-zero → the adaptive loop no-ops).
    coneGroup1 = device.createBindGroup({
      layout: conePipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: coneShader.shaderMeta.uniforms.tileHeader.binding,
          resource: { buffer: probeBufs.header },
        },
        {
          binding: coneShader.shaderMeta.uniforms.tileIndices.binding,
          resource: { buffer: probeBufs.indices },
        },
      ],
    });
  }

  // DOWNSCALED HDR target for the Layer-2 cone gather. coneScale = 2 (half-res, ¼ the pixels →
  // ~4× less cone work) by default; 4 (quarter-res, 1/16 the pixels) for heavily-loaded scenes.
  // The composite normal-aware-upsamples it back to full res (indirect light is low-frequency, so
  // that is fine). The render pass viewport IS the texture size, so the cone pass renders at the
  // downscaled res automatically; the cone shader maps via texCoord, so it needs no scale uniform.
  let coneScale = 2;
  const createConeOutput = () =>
    device.createTexture({
      size: [
        Math.max(1, Math.ceil(canvas.width / coneScale)),
        Math.max(1, Math.ceil(canvas.height / coneScale)),
        1,
      ],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  let coneOutput = createConeOutput();
  // Cached view; used both as the cone() attachment and as the composite's sampled binding.
  // Refreshed in resize() (the texture is rebuilt on resize, before the bind groups).
  let coneView = coneOutput.createView();

  // Upload the shared grid uniforms (populated in createVoxelSystem.buildGrid) to the cone shader.
  function uploadGridUniforms() {
    device.queue.writeBuffer(coneShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(coneShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
  }

  // VCT cone GI: the screen-probe RESOLVE (fill/bounce + emitter light via the probe SH) + short
  // AO cones → coneOutput (HALF-res HDR; composite bilinear-upsamples).
  // MUST run AFTER voxelize() + mips() + screenProbe(). Reads the G-buffer (depth + normal).
  function cone(encoder: GPUCommandEncoder) {
    // params/aoParams/tune are BAKED consts now — only params2 carries dynamic data.
    // .x = canvas width, .y = canvas height, .zw spare (the aniso toggle + light count ride the
    // gather's uLightParams).
    coneParams2Arr[0] = canvas.width;
    coneParams2Arr[1] = canvas.height;
    coneParams2Arr[2] = 0;
    coneParams2Arr[3] = 0;
    device.queue.writeBuffer(coneShader.uniforms.params2.getGPUBuffer(device), 0, coneParams2Arr);

    // params3: the screen-probe resolve params (all LIVE, GUI-tunable, no rebuild). .x = tile (the
    // resolve derives the probe grid from it), .y = normal-weight power, .z = plane-threshold scale,
    // .w = resolveRadius (the unified smooth kernel's support in tiles).
    coneParams3Arr[0] = screenProbe.getScreenProbeTile();
    coneParams3Arr[1] = screenProbe.getSpNormalPow();
    coneParams3Arr[2] = screenProbe.getSpPlaneK();
    coneParams3Arr[3] = screenProbe.getScreenProbeResolveRadius();
    device.queue.writeBuffer(coneShader.uniforms.params3.getGPUBuffer(device), 0, coneParams3Arr);

    // invViewProj computed ONCE per frame here (cone runs before composite, which reuses it).
    mat4.invert(invViewProj, viewProjMatrix);
    coneInvArr.set(invViewProj as Float32Array);
    device.queue.writeBuffer(coneShader.uniforms.invViewProj.getGPUBuffer(device), 0, coneInvArr);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: coneView,
          clearValue: [0, 0, 0, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(conePipeline);
    pass.setBindGroup(0, coneGroup0[screenProbe.getCurSet()]); // resolve reads THIS frame's atlas set
    pass.setBindGroup(1, coneGroup1); // adaptive-atlas tileHeader + tileIndices (StorageRead)
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // buildGrid hook: rebuild coneGroup0/1 from the fresh voxelRadiance view + re-sync the grid
  // uniforms (originArr/dimsArr already populated by createVoxelSystem.buildGrid this call).
  function rebindGrid() {
    buildConeGroup();
    uploadGridUniforms();
  }

  // Screen-probe onResourcesRecreated hook: rebuild coneGroup0 + coneGroup1 from the recreated atlas
  // (the cone resolve samples the recreated screen textures + rebinds tileHeader/tileIndices).
  function rebindGroups() {
    buildConeGroup();
  }

  // Change the cone-pass downscale factor (2 = half-res, 4 = quarter-res). Recreates the cone
  // output at the new size and fires onOutputRecreated (composite samples coneView). The cone
  // shader is resolution-agnostic; the composite reads coneScale via its params2 each frame.
  function setConeScale(newScale: number) {
    coneScale = Math.max(1, Math.round(newScale));
    coneOutput.destroy();
    coneOutput = createConeOutput();
    coneView = coneOutput.createView();
    onOutputRecreated();
  }

  // Canvas resized: recreate the canvas-sized cone output + rebuild the cone bind groups (which
  // reference the new G-buffer + screen textures). The composite group is rebuilt by the caller's
  // compositeSys.resize() (as in the original recreate()), so this does NOT fire onOutputRecreated.
  function resize() {
    coneOutput.destroy();
    coneOutput = createConeOutput();
    coneView = coneOutput.createView();
    buildConeGroup();
  }

  // Explicit, infrequent action: recompile the cone shader with the given config, recreate its
  // pipeline + bind groups, and re-upload the buildGrid-time grid uniforms that the fresh GPU
  // buffers lost (the per-frame ones refill next frame).
  function rebuild(newConfig: VoxelBakedConfig) {
    config = newConfig;
    coneShader.destroy();
    coneShader = new GPUShader(createConeShaderMeta(config));
    conePipeline = coneShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
      withBlending: false,
    });
    buildConeGroup();
    uploadGridUniforms();
  }

  return {
    cone,
    rebindGrid,
    rebindGroups,
    resize,
    rebuild,
    setConeScale,
    getOutputTexture: () => coneOutput,
    getOutputView: () => coneView,
    getConeScale: () => coneScale,
    getInvViewProj: () => invViewProj,
  };
}
