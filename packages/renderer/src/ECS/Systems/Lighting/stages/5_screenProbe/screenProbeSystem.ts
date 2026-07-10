// SCREEN-PROBE sub-system (the diffuse fill/bounce source: gather + temporal). Owns the
// screen-space probe atlas (SH-L1 ×3 + pixel/pos/nrm), the ping-pong temporal sets, the gather
// pipeline, and their bind groups + CPU scratch.
//
// The probe layout is the PLAIN UNIFORM GRID — one probe per SP_TILE² pixel tile, placed inline by
// the gather (tile center + stable hash jitter). The adaptive-density machinery that used to live
// here (light-adaptive sub-tile refinement + foveated rings: classify/decide/refine/args passes,
// the tile indirection buffers, the indirect adaptive gather, the budget readback) was REMOVED
// after profiling — the placement/indirection overhead ate the traced-cone savings, and the
// cone-output temporal filter smooths cheaper than extra probes.
import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { viewProjMatrix } from "../../../ResizeSystem.ts";
import { createScreenProbeShaderMeta, GATHER_WORKGROUP } from "./voxelScreenProbe.shader.ts";
import type { VoxelBakedConfig } from "../../core/voxelConfig.ts";
import { probeTile } from "../../core/voxelConfig.ts";
import {
  createScreenProbeTextures,
  destroyScreenProbeTextures,
  screenProbeGridDims,
  type ScreenProbeTextures,
} from "../../core/voxelResources.ts";
import type { createAnisoVolumeSystem } from "../4_anisoVolume/anisoVolumeSystem.ts";
import type { createEmitterLightsSystem } from "../../lights/emitterLightsSystem.ts";

export type ScreenProbeDeps = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  config: VoxelBakedConfig;
  // Shared grid-uniform scratch (populated by buildGrid before rebindGrid()/rebuild() run) —
  // buildScreenProbeGroups uploads these to the gather shader. Passed by reference (mutated in place
  // by the god file), so the module always sees the current grid values.
  originArr: ReturnType<typeof getTypeTypedArray>;
  dimsArr: ReturnType<typeof getTypeTypedArray>;
  // The grid voxelRadiance texture (gather group0 binds its all-mips 3d view). Read after buildGrid.
  getVoxelRadiance: () => GPUTexture;
  // The aniso sub-system — gather group0 binds its 6 directional volumes (getTextures()).
  aniso: ReturnType<typeof createAnisoVolumeSystem>;
  // The G-buffer (gather binds normal + depth). Read at group-build time.
  getGBuffer: () => { depth: GPUTexture; normal: GPUTexture };
  // The emitter-lights sub-system — gather group1 binds lightsBuf + clusterBuf.
  emitterLights: ReturnType<typeof createEmitterLightsSystem>;
  // Called after the atlas is recreated by rebuild() (the probe tile sizes it) — the god file
  // wires it to rebuild the CONE bind group (which samples the atlas).
  onResourcesRecreated: () => void;
  // Shared linear/mip filtering sampler for the voxelRadiance pyramid + aniso volumes (the SH atlas
  // textures are point-loaded, not sampled through this).
  voxelSampler: GPUSampler;
};

export function createScreenProbeSystem(deps: ScreenProbeDeps) {
  const {
    device,
    canvas,
    originArr,
    dimsArr,
    getVoxelRadiance,
    aniso,
    getGBuffer,
    emitterLights,
    voxelSampler,
    onResourcesRecreated,
  } = deps;
  // CURRENT baked config — the probe tile sizes the CPU atlas AND bakes into the shaders, so
  // rebuild(cfg) swaps this reference and recreates both from the same values.
  let config = deps.config;

  // Screen-space probe gather: the diffuse fill/bounce source. One thread per probe traces a
  // hemisphere of cones into voxelRadiance and writes SH-L1 (3 textures) + the probe's pixel/
  // validity + world anchor/normal. group0 = uniforms + G-buffer + voxelRadiance + aniso + sampler
  // + last frame's atlas as HISTORY; group1 = emitter lights + clusters; group2 = the 6 storage
  // outputs. DIRECT dispatch of ceil(numUniform / GATHER_WG) workgroups.
  let gatherShader = new GPUShader(createScreenProbeShaderMeta(config));
  let gatherPipeline = gatherShader.getComputePipeline(device, "main");

  // --- Scratch typed arrays for uniform uploads (every lane is live — the tunables are baked). ---
  const invViewProj = mat4.create(); // local inverse-viewProj for the gather uniform upload
  // Gather screenParams (.xy canvas, .z frame index) + the per-frame reverse-Z inverse-VP.
  const screenParamsArr = getTypeTypedArray(gatherShader.shaderMeta.uniforms.screenParams.type); // Float32Array(4)
  const screenInvArr = getTypeTypedArray(gatherShader.shaderMeta.uniforms.invViewProj.type); // Float32Array(16)
  // STAGE 3 (temporal) scratch. prevViewProjArr = LAST frame's forward viewProj, snapshotted at the
  // END of uploadProbeUniforms() (after this frame's uploads) — no other copy of viewProjMatrix is
  // retained across frames. Starts all-zero → the shader's prevClip.w <= 0 guard makes frame 1
  // fresh-only.
  const prevViewProjArr = getTypeTypedArray(gatherShader.shaderMeta.uniforms.prevViewProj.type); // Float32Array(16)

  // Screen-space probe textures (SH-L1 ×3 + pixel/pos/nrm). Resolution is CANVAS-derived (one probe
  // per tile) → recreated on resize / rebuild (the tile is baked config). STAGE 3 (temporal): TWO
  // full atlas sets, PING-PONGED by frame parity. spTex[curSet] is this frame's WRITE set (the
  // gather's group-2 storage targets AND what the cone-resolve/debug read this frame);
  // spTex[1 - curSet] is LAST frame's output = the HISTORY the gather reprojects from. The swap is
  // pure rebinding (two prebuilt bind-group variants per consumer, indexed by curSet — no per-frame
  // createBindGroup, no copies). Fresh textures are zero-filled → zero history normal → the
  // shader's validity test rejects them, so a recreate needs no explicit invalidation.
  let screenGrid = screenProbeGridDims(canvas.width, canvas.height, probeTile(config));
  let spTex: [ScreenProbeTextures, ScreenProbeTextures] = [
    createScreenProbeTextures(device, screenGrid),
    createScreenProbeTextures(device, screenGrid),
  ];

  // Frame counter → curSet = the ping-pong parity. Bumped ONCE per frame at the head of
  // uploadProbeUniforms(), so every later pass in the same frame sees one consistent parity. Also
  // rides screenParams.z (mod 1024) for the golden-angle cone-set rotation.
  let frameIndex = 0;
  let curSet = 0;

  // Gather bind groups: group0 (uniforms + G-buffer + voxelRadiance + aniso + sampler + the Stage-3
  // HISTORY views of the OTHER atlas set) is rebuilt when voxelRadiance changes (buildGrid) OR the
  // G-buffer / screen textures change (resize/rebuild); group1 (lightsBuf + clusterBuf) is rebuilt
  // when the emitter buffers grow; group2 (the 6 storage outputs) references the canvas-derived
  // spTex[parity].
  let gatherGroup0: [GPUBindGroup, GPUBindGroup];
  let gatherGroup1: GPUBindGroup;
  let gatherGroup2: [GPUBindGroup, GPUBindGroup];
  // (Re)build the gather bind groups + upload the (static-per-grid) gridOrigin/gridDims from the
  // arrays buildGrid has just populated (invViewProj/screenParams are dynamic → uploaded per frame).
  function buildScreenProbeGroups() {
    const { depth: gDepth, normal: gNormal } = getGBuffer();
    const gVoxelRadiance = getVoxelRadiance();
    // The 6 aniso directional volumes live in the aniso sub-system; buildGrid recreates them (via
    // anisoVolume.rebindGrid) BEFORE this runs, so read the fresh set here.
    const anisoTex = aniso.getTextures();
    // Parity variants: group0[pp] samples the OTHER set as HISTORY; group2[pp] storage-writes
    // spTex[pp]. group1 is parity-independent.
    const g0 = (pp: number) =>
      device.createBindGroup({
        layout: gatherPipeline.getBindGroupLayout(0),
        entries: [
          gatherShader.uniforms.gridOrigin.getBindGroupEntry(device),
          gatherShader.uniforms.gridDims.getBindGroupEntry(device),
          gatherShader.uniforms.invViewProj.getBindGroupEntry(device),
          gatherShader.uniforms.screenParams.getBindGroupEntry(device),
          // STAGE 3 temporal reprojection source (prev forward viewProj).
          gatherShader.uniforms.prevViewProj.getBindGroupEntry(device),
          // The 6 aniso volumes (ALL-mips views) — the gather owns the aimed cones + the
          // far-field anti-leak. (The emitter records are the uLights storage buffer in group 1.)
          {
            binding: gatherShader.shaderMeta.uniforms.anisoNegX.binding,
            resource: anisoTex!.negX.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.anisoPosX.binding,
            resource: anisoTex!.posX.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.anisoNegY.binding,
            resource: anisoTex!.negY.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.anisoPosY.binding,
            resource: anisoTex!.posY.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.anisoNegZ.binding,
            resource: anisoTex!.negZ.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.anisoPosZ.binding,
            resource: anisoTex!.posZ.createView({ dimension: "3d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.depthTex.binding,
            resource: gDepth.createView(),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.normalTex.binding,
            resource: gNormal.createView(),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.voxelRadiance.binding,
            resource: gVoxelRadiance.createView({ dimension: "3d" }),
          },
          { binding: gatherShader.shaderMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
          // STAGE 3 HISTORY: the OTHER atlas set (last frame's group-2 output) as sampled views —
          // read-only history rides group 0, keeping the group-2 storage-texture budget untouched.
          {
            binding: gatherShader.shaderMeta.uniforms.histShR.binding,
            resource: spTex[1 - pp].shR.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.histShG.binding,
            resource: spTex[1 - pp].shG.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.histShB.binding,
            resource: spTex[1 - pp].shB.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.histPos.binding,
            resource: spTex[1 - pp].pos.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.histNrm.binding,
            resource: spTex[1 - pp].nrm.createView({ dimension: "2d" }),
          },
        ],
      });
    gatherGroup1 = device.createBindGroup({
      layout: gatherPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: gatherShader.shaderMeta.uniforms.lightsData.binding,
          resource: { buffer: emitterLights.lightsBuf },
        },
        {
          binding: gatherShader.shaderMeta.uniforms.lightClusters.binding,
          resource: { buffer: emitterLights.clusterBuf! },
        },
      ],
    });
    const g2 = (pp: number) =>
      device.createBindGroup({
        layout: gatherPipeline.getBindGroupLayout(2),
        entries: [
          {
            binding: gatherShader.shaderMeta.uniforms.screenShR.binding,
            resource: spTex[pp].shR.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.screenShG.binding,
            resource: spTex[pp].shG.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.screenShB.binding,
            resource: spTex[pp].shB.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.screenProbePix.binding,
            resource: spTex[pp].pix.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.screenProbePos.binding,
            resource: spTex[pp].pos.createView({ dimension: "2d" }),
          },
          {
            binding: gatherShader.shaderMeta.uniforms.screenProbeNrm.binding,
            resource: spTex[pp].nrm.createView({ dimension: "2d" }),
          },
        ],
      });
    gatherGroup0 = [g0(0), g0(1)];
    gatherGroup2 = [g2(0), g2(1)];
    // Shares the SAME world box as the grid (origin + cellSize + voxel dims). originArr/dimsArr are
    // populated by buildGrid before this runs; on rebuild() they are re-set from the same values.
    device.queue.writeBuffer(gatherShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    device.queue.writeBuffer(gatherShader.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
  }

  // THE probe pass: upload the per-frame uniforms, tick the ping-pong parity, and gather every
  // probe of the uniform grid in ONE direct dispatch. MUST run AFTER mips() (the gather samples
  // the pyramid) and BEFORE cone() (the resolve reads this frame's atlas).
  function gatherProbes(encoder: GPUCommandEncoder) {
    // STAGE 3: one frame tick — flips the ping-pong parity for EVERY consumer this frame.
    frameIndex++;
    curSet = frameIndex & 1;
    mat4.invert(invViewProj, viewProjMatrix);
    screenInvArr.set(invViewProj as Float32Array);
    // Canvas + frame index (mod 1024 for f32 exactness — the golden-angle rotation + the aimed
    // round-robin seed); prevViewProj = LAST frame's forward matrix (snapshotted below).
    screenParamsArr[0] = canvas.width;
    screenParamsArr[1] = canvas.height;
    screenParamsArr[2] = frameIndex % 1024;
    device.queue.writeBuffer(
      gatherShader.uniforms.screenParams.getGPUBuffer(device),
      0,
      screenParamsArr,
    );
    device.queue.writeBuffer(gatherShader.uniforms.invViewProj.getGPUBuffer(device), 0, screenInvArr);
    device.queue.writeBuffer(
      gatherShader.uniforms.prevViewProj.getGPUBuffer(device),
      0,
      prevViewProjArr,
    );
    // Snapshot THIS frame's forward viewProj for next frame's reprojection (the only retained copy).
    prevViewProjArr.set(viewProjMatrix as Float32Array);

    const numProbes = screenGrid.w * screenGrid.h;
    const pass = encoder.beginComputePass();
    pass.setPipeline(gatherPipeline);
    pass.setBindGroup(0, gatherGroup0[curSet]); // history = spTex[1 - curSet] sampled views
    pass.setBindGroup(1, gatherGroup1);
    pass.setBindGroup(2, gatherGroup2[curSet]); // storage-writes spTex[curSet]
    pass.dispatchWorkgroups(Math.ceil(numProbes / GATHER_WORKGROUP), 1, 1);
    pass.end();
  }

  // Recreate the canvas-derived probe atlas (both ping-pong sets — a stale-sized history can never
  // be sampled; fresh zero-filled textures auto-invalidate as history). Called on resize / rebuild.
  function recreateScreenProbeResources() {
    screenGrid = screenProbeGridDims(canvas.width, canvas.height, probeTile(config));
    destroyScreenProbeTextures(spTex[0]);
    destroyScreenProbeTextures(spTex[1]);
    spTex = [
      createScreenProbeTextures(device, screenGrid),
      createScreenProbeTextures(device, screenGrid),
    ];
  }

  // Grid rebuilt (buildGrid): rebuild the gather groups from the fresh voxelRadiance + aniso
  // volumes (+ re-upload the gather grid uniforms).
  function rebindGrid() {
    buildScreenProbeGroups();
  }

  // emitterLights' onBuffersRecreated (lightsBuf grow): rebuild the gather group1 (binds lightsBuf).
  function rebuildGroups() {
    buildScreenProbeGroups();
  }

  // Canvas resized: recreate the canvas-derived atlas, then rebuild the groups.
  function resize() {
    recreateScreenProbeResources();
    buildScreenProbeGroups();
  }

  // Explicit rebuild (config change): recompile the gather shader with the CURRENT config
  // (tile / hysteresis / bilateral weights / aniso mode / aimed group), recreate the tile-sized
  // atlas from the same config, rebuild the groups, and fire onResourcesRecreated (the cone
  // resolve samples the recreated atlas).
  function rebuild(cfg: VoxelBakedConfig) {
    config = cfg;
    gatherShader.destroy();
    gatherShader = new GPUShader(createScreenProbeShaderMeta(cfg));
    gatherPipeline = gatherShader.getComputePipeline(device, "main");
    recreateScreenProbeResources();
    buildScreenProbeGroups();
    onResourcesRecreated();
  }

  // Camera-following grid: re-upload ONLY uGridOrigin (originArr refreshed by the caller — carries
  // the zoom-ladder cellSize in .w too). A uniform write — dims/textures unchanged, no rebuild.
  function uploadGridOrigin() {
    device.queue.writeBuffer(gatherShader.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
  }

  return {
    // Pass methods.
    gatherProbes,
    // Lifecycle.
    rebindGrid,
    rebuildGroups,
    resize,
    rebuild,
    uploadGridOrigin,
    // For the CONE cluster (still in createVoxelSystem, consumes these).
    getSpTex: () => spTex,
    getCurSet: () => curSet,
  };
}
