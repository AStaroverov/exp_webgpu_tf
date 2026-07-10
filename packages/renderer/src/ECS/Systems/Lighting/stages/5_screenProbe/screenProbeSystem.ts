// SCREEN-PROBE sub-system (the diffuse fill/bounce source: gather + adaptive placement + temporal +
// budget + debug). Owns the screen-space probe atlas (SH-L1 ×3 + pixel/validity), the ping-pong
// temporal sets, the adaptive-atlas indirection buffers, the two gather pipelines (uniform + adaptive)
// + the classify/refine/args placement pipelines + the debug view, and all their bind groups + CPU
// scratch. Extracted VERBATIM from createVoxelSystem — behavior is byte-for-byte identical.
import { mat4 } from "gl-matrix";
import { GPUShader } from "../../../../../WGSL/GPUShader.ts";
import { getTypeTypedArray } from "../../../../../Shader/index.ts";
import { viewProjMatrix } from "../../../ResizeSystem.ts";
import { createScreenProbeShaderMeta, GATHER_WORKGROUP } from "./voxelScreenProbe.shader.ts";
import { createClassifyShaderMeta, WORKGROUP as CLASSIFY_WG } from "./voxelProbeClassify.shader.ts";
import { createDecideShaderMeta, DECIDE_WORKGROUP as DECIDE_WG } from "./voxelProbeDecide.shader.ts";
import { createRefineShaderMeta, WORKGROUP as REFINE_WG } from "./voxelProbeRefine.shader.ts";
import { argsShaderMeta } from "./voxelProbeArgs.shader.ts";
import { createDebugShaderMeta } from "./voxelProbeDebug.shader.ts";
import type { VoxelBakedConfig } from "../../core/voxelConfig.ts";
import {
  createScreenProbeBuffers,
  createScreenProbeTextures,
  destroyScreenProbeBuffers,
  destroyScreenProbeTextures,
  screenProbeCounts,
  screenProbeGridDims,
  type ScreenProbeBuffers,
  type ScreenProbeCounts,
  type ScreenProbeTextures,
} from "../../core/voxelResources.ts";
import { probeRefineDiv, probeTile } from "../../core/voxelConfig.ts";
import type { createAnisoVolumeSystem } from "../4_anisoVolume/anisoVolumeSystem.ts";
import type { createEmitterLightsSystem } from "../../lights/emitterLightsSystem.ts";

export type ScreenProbeDeps = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  config: VoxelBakedConfig;
  // Shared grid-uniform scratch (populated by buildGrid before rebindGrid()/rebuild() run) —
  // buildScreenProbeGroups uploads these to the gather shaders. Passed by reference (mutated in place
  // by the god file), so the module always sees the current grid values.
  originArr: ReturnType<typeof getTypeTypedArray>;
  dimsArr: ReturnType<typeof getTypeTypedArray>;
  // The grid voxelRadiance texture (gather group0 binds its all-mips 3d view). Read after buildGrid.
  getVoxelRadiance: () => GPUTexture;
  // The aniso sub-system — gather group0 binds its 6 directional volumes (getTextures()).
  aniso: ReturnType<typeof createAnisoVolumeSystem>;
  // The G-buffer (gather/refine bind normal + depth; debug binds normal). Read at group-build time.
  getGBuffer: () => { depth: GPUTexture; normal: GPUTexture };
  // The emitter-lights sub-system — gather group1 binds lightsBuf + clusterBuf; uploadProbeUniforms
  // reads getLightCount().
  emitterLights: ReturnType<typeof createEmitterLightsSystem>;
  // probeDebug renders into the composite output view (= compositeSys.getOutputView()).
  getDebugTargetView: () => GPUTextureView;
  // Called after resources are recreated by setScreenProbeTile / setAdaptiveFraction — the god file
  // wires it to rebuild the CONE bind group (not yet extracted).
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
    getDebugTargetView,
    voxelSampler,
    onResourcesRecreated,
  } = deps;
  // CURRENT baked config — the probe tile + refine divisor size the CPU resources AND bake into
  // the shaders, so rebuild(cfg) swaps this reference and recreates both from the same values.
  let config = deps.config;

  // Screen-space probe gather: the diffuse fill/bounce source. One thread per screen probe traces a
  // hemisphere of cones into voxelRadiance (verbatim probe math) and writes SH-L1 (3 textures) + the
  // probe's representative pixel/validity (1 texture). group0 = uniforms + G-buffer + voxelRadiance +
  // sampler, group1 = probeData + counter, group2 = the 4 storage outputs. TWO pipelines from the one
  // factory (IS_ADAPTIVE baked, identical bindings): gatherUniformPipeline (direct dispatch over the
  // uniform block, runs BEFORE refine so refine reads its raw SH) and gatherAdaptivePipeline (indirect
  // dispatch over the refine-placed adaptive tail, runs AFTER buildArgs). Both write the SAME atlas.
  let gatherUniformShader = new GPUShader(createScreenProbeShaderMeta(config, false));
  let gatherUniformPipeline = gatherUniformShader.getComputePipeline(device, "main");
  let gatherAdaptiveShader = new GPUShader(createScreenProbeShaderMeta(config, true));
  let gatherAdaptivePipeline = gatherAdaptiveShader.getComputePipeline(device, "main");

  // Adaptive-atlas placement passes. classify/decide/refine/debug BAKE config tunables (the
  // foveated ring thresholds; decide also the light trigger), so — like the gathers — they are
  // `let`s recompiled by rebuild(). Only args (pure counter math, no tunables) is built once.
  // classify (uniform placement) uses groups 0 (uniforms) + 1 (refineState, read) + 2 (probeData);
  // refine uses 0 (uniforms + G-buffer) + 1 (refineState, read) + 2 (allocator buffers).
  let classifyShader = new GPUShader(createClassifyShaderMeta(config));
  let classifyPipeline = classifyShader.getComputePipeline(device, "main");
  // REFINEMENT DECISION pass (one thread per coarse tile): the raw light trigger + the persistent
  // per-tile hysteresis counter live here — the refine pass only executes the settled decision.
  let decideShader = new GPUShader(createDecideShaderMeta(config));
  let decidePipeline = decideShader.getComputePipeline(device, "main");
  // The refine (spawn) pass: one thread per fine cell of a REFINED tile (cell divisor refineDiv1 →
  // screenParams.w) spawns the tile's full fine lattice. group 1 = refineState (read).
  let refineShader = new GPUShader(createRefineShaderMeta(config));
  let refinePipeline = refineShader.getComputePipeline(device, "main");
  const argsShader = new GPUShader(argsShaderMeta);
  const argsPipeline = argsShader.getComputePipeline(device, "main");

  // Screen-probe DEBUG view: a fullscreen pass that REPLACES the composite when debugProbes is on,
  // false-coloring the probe distribution (uniform grid + adaptive probes + subdivided tiles).
  let debugShader = new GPUShader(createDebugShaderMeta(config));
  let debugPipeline = debugShader.getRenderPipeline(device, "vs_main", "fs_main", {
    targetFormat: "rgba16float",
    withBlending: false,
  });

  // --- Scratch typed arrays for uniform uploads (every lane is live — the tunables are baked). ---
  const invViewProj = mat4.create(); // local inverse-viewProj for the gather uniform upload
  // Gather screenParams (.xy canvas, .z frame index, .w maxAdaptive) + the per-frame reverse-Z
  // inverse-VP. Types are identical across both gather variants → read from the uniform one.
  const screenParamsArr = getTypeTypedArray(
    gatherUniformShader.shaderMeta.uniforms.screenParams.type,
  ); // Float32Array(4)
  const screenInvArr = getTypeTypedArray(gatherUniformShader.shaderMeta.uniforms.invViewProj.type); // Float32Array(16)
  // STAGE 3 (temporal) scratch. prevViewProjArr = LAST frame's forward viewProj, snapshotted at the
  // END of uploadProbeUniforms() (after this frame's uploads) — no other copy of viewProjMatrix is
  // retained across frames. Starts all-zero → the shader's prevClip.w <= 0 guard makes frame 1
  // fresh-only.
  const prevViewProjArr = getTypeTypedArray(
    gatherUniformShader.shaderMeta.uniforms.prevViewProj.type,
  ); // Float32Array(16)
  // Placement/debug-pass scratch (allocated ONCE): canvas dims (vec2) for classify/decide/debug;
  // refine adds maxAdaptive (.z); args = (numUniform, maxAdaptive).
  const classifyScreenArr = new Float32Array(2);
  const decideScreenArr = new Float32Array(2);
  const debugParamsArr = new Float32Array(2);
  const refineScreenArr = new Float32Array(4);
  const argsParamsArr = new Uint32Array(4);

  // Screen-space probe textures (SH-L1 ×3 + pixel/validity ×1). Resolution is CANVAS-derived
  // (one probe per SCREEN_PROBE_TILE² tile) → created here + recreated on resize (recreate()),
  // exactly like coneOutput. `let` so recreate() can reassign the whole set.
  let screenGrid = screenProbeGridDims(canvas.width, canvas.height, probeTile(config));
  // ADAPTIVE screen-probe atlas. adaptiveFraction sizes the atlas + buffers for numUniform +
  // maxAdaptive probes (default 1.0 → 2× uniform, no temporal amortization). The adaptive block
  // occupies atlas rows [gh, gh + adaptiveRows). The single refine pass spawns light-adaptively:
  // large lightThresh ⇒ no adaptive probes (the flat uniform atlas); lower ⇒ denser where the
  // gathered light varies.
  let adaptiveFraction = 1.0;
  let probeCounts: ScreenProbeCounts = screenProbeCounts(
    screenGrid,
    adaptiveFraction,
    probeRefineDiv(config),
  );
  // STAGE 3 (temporal): TWO full atlas sets, PING-PONGED by frame parity. spTex[curSet] is this
  // frame's WRITE set (the gather's group-2 storage targets AND what refine/cone-resolve/debug read
  // this frame); spTex[1 - curSet] is LAST frame's output = the HISTORY the gather reprojects from.
  // The swap is pure rebinding (two prebuilt bind-group variants per consumer, indexed by curSet —
  // no per-frame createBindGroup, no copies). Fresh textures are zero-filled → zero history normal
  // → the shader's validity test rejects them, so a recreate needs no explicit invalidation.
  let spTex: [ScreenProbeTextures, ScreenProbeTextures] = [
    createScreenProbeTextures(device, screenGrid, probeCounts.adaptiveRows),
    createScreenProbeTextures(device, screenGrid, probeCounts.adaptiveRows),
  ];
  // The indirection + allocator buffers (RAW, shared across the classify/refine/args/gather/resolve
  // passes → bound manually at each shader's declared binding). Recreated wherever screenProbeTex is.
  let probeBufs: ScreenProbeBuffers = createScreenProbeBuffers(device, probeCounts);

  // (The probe tile + refine divisor live in the BAKED config — probeTile(config) /
  // probeRefineDiv(config) — like every other tuning knob; rebuild(cfg) recreates the resources
  // they size. adaptiveFraction stays runtime state: it only sizes buffers, no shader reads it.)
  // Frame counter → curSet = the ping-pong parity. Bumped ONCE per frame at the head of
  // uploadProbeUniforms() (pass A0), so every later pass in the same frame sees one consistent
  // parity. Also rides uTemporalParams.y (mod 1024) for the golden-angle cone-set rotation.
  let frameIndex = 0;
  let curSet = 0;
  // Debug view: when on, probeDebug() replaces composite() with a false-color map of the probe
  // distribution (uniform grid + adaptive probes + subdivided tiles). Live GUI toggle, no rebuild.
  let debugProbes = false;

  // Gather bind groups: group0 (uniforms + G-buffer + voxelRadiance all-mips view + sampler + the
  // Stage-3 HISTORY views of the OTHER atlas set) is rebuilt when voxelRadiance changes (buildGrid)
  // OR the G-buffer / screen textures change (resize); group1 (probeData + counter) references the
  // RAW probeBufs (parity-independent → single); group2 (the 6 storage outputs) references the
  // canvas-derived spTex[parity]. The uniform + adaptive gather pipelines have IDENTICAL bindings
  // but SEPARATE uniform buffers (own GPUShader) → one group set per pipeline.
  let gatherUniformGroup0: [GPUBindGroup, GPUBindGroup];
  let gatherUniformGroup1: GPUBindGroup;
  let gatherUniformGroup2: [GPUBindGroup, GPUBindGroup];
  let gatherAdaptiveGroup0: [GPUBindGroup, GPUBindGroup];
  let gatherAdaptiveGroup1: GPUBindGroup;
  let gatherAdaptiveGroup2: [GPUBindGroup, GPUBindGroup];
  // Adaptive-atlas placement/indirection bind groups (rebuilt alongside the screen-probe groups):
  // the classify/refine/args groups. All reference the RAW probeBufs, so they are rebuilt whenever
  // probeBufs is recreated (resize / tile / adaptiveFraction change). The gather's group 1 lives in
  // the per-pipeline sets declared above.
  let classifyGroup0: GPUBindGroup;
  let classifyGroup1: GPUBindGroup; // refineState (read) — the foveated boost
  let classifyGroup2: GPUBindGroup;
  // DECIDE pass: group0 = uniforms + G-buffer normal + raw uniform SH (reads the SH the uniform
  // gather wrote THIS frame → its inSh* bind the CURRENT set, per parity); group1 = probeData
  // (read); group2 = refineState (read_write, persistent).
  let decideGroup0: [GPUBindGroup, GPUBindGroup];
  let decideGroup1: GPUBindGroup;
  let decideGroup2: GPUBindGroup;
  // REFINE (spawn) pass: group0 = uniforms + G-buffer normal (parity-independent now — the light
  // trigger moved to decide); group1 = refineState (read); group2 = the allocator buffers.
  let refineGroup0: GPUBindGroup;
  let refineGroup1: GPUBindGroup;
  let refineGroup2: GPUBindGroup;
  let argsGroup0: GPUBindGroup;
  let argsGroup1: GPUBindGroup;
  let argsGroup2: GPUBindGroup;
  let debugGroup0: [GPUBindGroup, GPUBindGroup];
  let debugGroup1: GPUBindGroup;

  // (Re)build the debug-view bind group: group0 = params uniform + normal + screenProbePix; group1 =
  // tileHeader + tileIndices (StorageRead). Called from buildScreenProbeGroups so it tracks every
  // recreate.
  function buildProbeDebugGroup() {
    const { normal: gNormal } = getGBuffer();
    // Per-parity (pix of THIS frame's write set).
    const buildDebugGroup0 = (tex: ScreenProbeTextures) =>
      device.createBindGroup({
        layout: debugPipeline.getBindGroupLayout(0),
        entries: [
          debugShader.uniforms.params.getBindGroupEntry(device),
          {
            binding: debugShader.shaderMeta.uniforms.normalTex.binding,
            resource: gNormal.createView(),
          },
          {
            binding: debugShader.shaderMeta.uniforms.screenProbePix.binding,
            resource: tex.pix.createView({ dimension: "2d" }),
          },
        ],
      });
    debugGroup0 = [buildDebugGroup0(spTex[0]), buildDebugGroup0(spTex[1])];
    debugGroup1 = device.createBindGroup({
      layout: debugPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: debugShader.shaderMeta.uniforms.tileHeader.binding,
          resource: { buffer: probeBufs.header },
        },
        {
          binding: debugShader.shaderMeta.uniforms.tileIndices.binding,
          resource: { buffer: probeBufs.indices },
        },
      ],
    });
  }

  // (Re)build the screen-probe bind groups: group0 = uniforms + the G-buffer (depth/normal) + the
  // ALL-mips voxelRadiance view + the sampler; group2 = the 4 storage outputs (SH ×3 + pixel). group0
  // references voxelRadiance (rebuilt on grid change) AND the G-buffer + screen textures (rebuilt on
  // resize), so this is called from buildGrid AND recreate. Also uploads the (static-per-grid)
  // gridOrigin/gridDims from the arrays buildGrid has just populated (invViewProj/screenParams are
  // dynamic → uploaded per frame in screenProbe()).
  function buildScreenProbeGroups() {
    const { depth: gDepth, normal: gNormal } = getGBuffer();
    const gVoxelRadiance = getVoxelRadiance();
    // The 6 aniso directional volumes live in the aniso sub-system; buildGrid recreates them (via
    // anisoVolume.rebindGrid) BEFORE this runs, so read the fresh set here.
    const anisoTex = aniso.getTextures();
    // One (group0, group1, group2) set per gather pipeline. The uniform + adaptive variants have
    // IDENTICAL bindings but own uniform buffers (separate GPUShader), so build both from one helper
    // (mirrors buildRefineGroups). group0 = uniforms + G-buffer + voxelRadiance + sampler; group1 =
    // probeData + counter (RAW probeBufs); group2 = the 4 storage outputs (SH ×3 + pixel).
    const buildGatherGroups = (
      shader: typeof gatherUniformShader,
      pipeline: GPUComputePipeline,
    ): {
      g0: [GPUBindGroup, GPUBindGroup];
      g1: GPUBindGroup;
      g2: [GPUBindGroup, GPUBindGroup];
    } => {
      // Parity variants: group0[pp] samples the OTHER set as HISTORY; group2[pp] storage-writes
      // spTex[pp]. group1 (RAW probeBufs) is parity-independent.
      const g0 = (pp: number) =>
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            shader.uniforms.gridOrigin.getBindGroupEntry(device),
            shader.uniforms.gridDims.getBindGroupEntry(device),
            shader.uniforms.invViewProj.getBindGroupEntry(device),
            shader.uniforms.screenParams.getBindGroupEntry(device),
            // STAGE 3 temporal reprojection source (prev forward viewProj).
            shader.uniforms.prevViewProj.getBindGroupEntry(device),
            // The 6 aniso volumes (ALL-mips views) — the gather owns the aimed cones + the
            // far-field anti-leak. (The emitter records are the uLights storage buffer in group 1.)
            {
              binding: shader.shaderMeta.uniforms.anisoNegX.binding,
              resource: anisoTex!.negX.createView({ dimension: "3d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.anisoPosX.binding,
              resource: anisoTex!.posX.createView({ dimension: "3d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.anisoNegY.binding,
              resource: anisoTex!.negY.createView({ dimension: "3d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.anisoPosY.binding,
              resource: anisoTex!.posY.createView({ dimension: "3d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.anisoNegZ.binding,
              resource: anisoTex!.negZ.createView({ dimension: "3d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.anisoPosZ.binding,
              resource: anisoTex!.posZ.createView({ dimension: "3d" }),
            },
            { binding: shader.shaderMeta.uniforms.depthTex.binding, resource: gDepth.createView() },
            {
              binding: shader.shaderMeta.uniforms.normalTex.binding,
              resource: gNormal.createView(),
            },
            {
              binding: shader.shaderMeta.uniforms.voxelRadiance.binding,
              resource: gVoxelRadiance.createView({ dimension: "3d" }),
            },
            { binding: shader.shaderMeta.uniforms.voxelSampler.binding, resource: voxelSampler },
            // STAGE 3 HISTORY: the OTHER atlas set (last frame's group-2 output) as sampled views —
            // read-only history rides group 0, keeping the group-2 storage-texture budget untouched.
            {
              binding: shader.shaderMeta.uniforms.histShR.binding,
              resource: spTex[1 - pp].shR.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.histShG.binding,
              resource: spTex[1 - pp].shG.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.histShB.binding,
              resource: spTex[1 - pp].shB.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.histPos.binding,
              resource: spTex[1 - pp].pos.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.histNrm.binding,
              resource: spTex[1 - pp].nrm.createView({ dimension: "2d" }),
            },
          ],
        });
      const g1 = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: shader.shaderMeta.uniforms.probeData.binding,
            resource: { buffer: probeBufs.data },
          },
          {
            binding: shader.shaderMeta.uniforms.probeCounter.binding,
            resource: { buffer: probeBufs.counter },
          },
          {
            binding: shader.shaderMeta.uniforms.activeList.binding,
            resource: { buffer: probeBufs.activeList },
          },
          {
            binding: shader.shaderMeta.uniforms.lightsData.binding,
            resource: { buffer: emitterLights.lightsBuf },
          },
          {
            binding: shader.shaderMeta.uniforms.lightClusters.binding,
            resource: { buffer: emitterLights.clusterBuf! },
          },
        ],
      });
      const g2 = (pp: number) =>
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(2),
          entries: [
            {
              binding: shader.shaderMeta.uniforms.screenShR.binding,
              resource: spTex[pp].shR.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.screenShG.binding,
              resource: spTex[pp].shG.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.screenShB.binding,
              resource: spTex[pp].shB.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.screenProbePix.binding,
              resource: spTex[pp].pix.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.screenProbePos.binding,
              resource: spTex[pp].pos.createView({ dimension: "2d" }),
            },
            {
              binding: shader.shaderMeta.uniforms.screenProbeNrm.binding,
              resource: spTex[pp].nrm.createView({ dimension: "2d" }),
            },
          ],
        });
      return { g0: [g0(0), g0(1)], g1, g2: [g2(0), g2(1)] };
    };
    {
      const u = buildGatherGroups(gatherUniformShader, gatherUniformPipeline);
      gatherUniformGroup0 = u.g0;
      gatherUniformGroup1 = u.g1;
      gatherUniformGroup2 = u.g2;
      const a = buildGatherGroups(gatherAdaptiveShader, gatherAdaptivePipeline);
      gatherAdaptiveGroup0 = a.g0;
      gatherAdaptiveGroup1 = a.g1;
      gatherAdaptiveGroup2 = a.g2;
    }
    // Shares the SAME world box as the grid (origin + cellSize + voxel dims). originArr/dimsArr are
    // populated by buildGrid before this runs; on rebuild() they are re-set from the same values.
    // Uploaded to BOTH gather shaders (each has its own uniform buffers).
    for (const s of [gatherUniformShader, gatherAdaptiveShader]) {
      device.queue.writeBuffer(s.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
      device.queue.writeBuffer(s.uniforms.gridDims.getGPUBuffer(device), 0, dimsArr);
    }

    // ===== Adaptive-atlas placement bind groups (all reference the RAW shared probeBufs). =====
    // (The gather's group 1 = probeData + counter is built per-pipeline in buildGatherGroups above.)
    // Classify (Pass A0): group 0 = screenParams uniform (no G-buffer — pure center+jitter
    // placement; rings are baked); group 1 = refineState (read — the foveated boost); group 2 =
    // probeData.
    classifyGroup0 = device.createBindGroup({
      layout: classifyPipeline.getBindGroupLayout(0),
      entries: [classifyShader.uniforms.screenParams.getBindGroupEntry(device)],
    });
    classifyGroup1 = device.createBindGroup({
      layout: classifyPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: classifyShader.shaderMeta.uniforms.refineState.binding,
          resource: { buffer: probeBufs.refineState },
        },
      ],
    });
    classifyGroup2 = device.createBindGroup({
      layout: classifyPipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: classifyShader.shaderMeta.uniforms.probeData.binding,
          resource: { buffer: probeBufs.data },
        },
      ],
    });
    // DECIDE (per-tile refinement decision): group 0 = uniforms + G-buffer normal + the RAW
    // uniform SH atlas (point-loaded for the radiometric trigger — what gatherUniform wrote just
    // before, THIS frame → per parity, bound to the CURRENT write set, never history); group 1 =
    // probeData (cage validity); group 2 = the persistent refineState.
    const buildDecideGroup0 = (tex: ScreenProbeTextures) =>
      device.createBindGroup({
        layout: decidePipeline.getBindGroupLayout(0),
        entries: [
          decideShader.uniforms.screenParams.getBindGroupEntry(device),
          {
            binding: decideShader.shaderMeta.uniforms.normalTex.binding,
            resource: gNormal.createView(),
          },
          {
            binding: decideShader.shaderMeta.uniforms.inShR.binding,
            resource: tex.shR.createView({ dimension: "2d" }),
          },
          {
            binding: decideShader.shaderMeta.uniforms.inShG.binding,
            resource: tex.shG.createView({ dimension: "2d" }),
          },
          {
            binding: decideShader.shaderMeta.uniforms.inShB.binding,
            resource: tex.shB.createView({ dimension: "2d" }),
          },
        ],
      });
    decideGroup0 = [buildDecideGroup0(spTex[0]), buildDecideGroup0(spTex[1])];
    decideGroup1 = device.createBindGroup({
      layout: decidePipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: decideShader.shaderMeta.uniforms.probeData.binding,
          resource: { buffer: probeBufs.data },
        },
      ],
    });
    decideGroup2 = device.createBindGroup({
      layout: decidePipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: decideShader.shaderMeta.uniforms.refineState.binding,
          resource: { buffer: probeBufs.refineState },
        },
      ],
    });
    // Refine (spawn): group 0 = uniforms + G-buffer normal (parity-independent — the trigger moved
    // to decide); group 1 = refineState (read); group 2 = the allocator buffers (RAW shared probeBufs).
    refineGroup0 = device.createBindGroup({
      layout: refinePipeline.getBindGroupLayout(0),
      entries: [
        refineShader.uniforms.screenParams.getBindGroupEntry(device),
        {
          binding: refineShader.shaderMeta.uniforms.normalTex.binding,
          resource: gNormal.createView(),
        },
      ],
    });
    refineGroup1 = device.createBindGroup({
      layout: refinePipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: refineShader.shaderMeta.uniforms.refineState.binding,
          resource: { buffer: probeBufs.refineState },
        },
      ],
    });
    refineGroup2 = device.createBindGroup({
      layout: refinePipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: refineShader.shaderMeta.uniforms.probeCounter.binding,
          resource: { buffer: probeBufs.counter },
        },
        {
          binding: refineShader.shaderMeta.uniforms.probeData.binding,
          resource: { buffer: probeBufs.data },
        },
        {
          binding: refineShader.shaderMeta.uniforms.tileHeader.binding,
          resource: { buffer: probeBufs.header },
        },
        {
          binding: refineShader.shaderMeta.uniforms.tileIndices.binding,
          resource: { buffer: probeBufs.indices },
        },
        {
          binding: refineShader.shaderMeta.uniforms.activeList.binding,
          resource: { buffer: probeBufs.activeList },
        },
      ],
    });
    // Args (Pass B): group 0 = params uniform; group 1 = counter (read); group 2 = the RAW indirect
    // args buffer (the one bit GPUVariable can't express → substitute our own INDIRECT-capable buffer).
    argsGroup0 = device.createBindGroup({
      layout: argsPipeline.getBindGroupLayout(0),
      entries: [argsShader.uniforms.params.getBindGroupEntry(device)],
    });
    argsGroup1 = device.createBindGroup({
      layout: argsPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: argsShader.shaderMeta.uniforms.counter.binding,
          resource: { buffer: probeBufs.counter },
        },
      ],
    });
    argsGroup2 = device.createBindGroup({
      layout: argsPipeline.getBindGroupLayout(2),
      entries: [
        {
          binding: argsShader.shaderMeta.uniforms.args.binding,
          resource: { buffer: probeBufs.args },
        },
      ],
    });
    // (The refine pass no longer has grid uniforms — the light-adaptive trigger needs no world-space
    // reconstruction, only the screen-space cage SH and the G-buffer normal.)

    // The debug view reads the same G-buffer normal + atlas pix + indirection; rebuild it alongside
    // the probe groups (both depend on gNormal / screenProbeTex / probeBufs → same recreate triggers).
    buildProbeDebugGroup();
  }

  // ===== Adaptive screen-probe atlas pass chain (light-adaptive density). =====
  // Frame order: probeClear → probeClassify → gatherUniform → probeRefine (16→8) → probeBuildArgs →
  // gatherAdaptive → cone. The single gather is split: gatherUniform runs BEFORE refine (a DIRECT
  // dispatch over the uniform block) so refine can read its raw SH as the light-adaptive subdivision
  // signal; gatherAdaptive runs AFTER buildArgs (INDIRECT over the refine-placed adaptive tail). Both
  // write the SAME raw SH atlas. Each compute pass is its OWN beginComputePass/end so the encoder
  // barriers between them (classify's probeData → gatherUniform → refine reads its SH; refine's
  // counter → args → the indirect gatherAdaptive — the SAME implicit inter-pass barrier the mip/aniso
  // chains rely on). MUST run AFTER mips() (the gather samples the pyramid) and BEFORE cone() (resolve).

  // Pass 0 — Clear (host-side, no shader). Zero the allocator counter (+ budget flag), the per-tile
  // adaptive counts, and the indirect args. probeData/tileIndices need no clear (only written slots
  // are ever read, gated by the counts). This explicit first step IS the anti-staleness mechanism.
  function probeClear(encoder: GPUCommandEncoder) {
    encoder.clearBuffer(probeBufs.counter);
    encoder.clearBuffer(probeBufs.header);
    encoder.clearBuffer(probeBufs.args);
  }

  // Upload the per-frame uniforms for the WHOLE chain (queue writes all land before the encoder runs,
  // so doing them here at the head is correct regardless of pass order). invViewProj is the reverse-Z
  // inverse-VP; cone() recomputes it right after into the same scratch — harmless.
  function uploadProbeUniforms() {
    // STAGE 3: one frame tick — flips the ping-pong parity for EVERY pass this frame (this runs
    // at the head of the chain, pass A0, before any atlas-referencing dispatch).
    frameIndex++;
    curSet = frameIndex & 1;
    mat4.invert(invViewProj, viewProjMatrix);
    screenInvArr.set(invViewProj as Float32Array);
    // classify/decide: canvas dims only (every tunable is baked).
    classifyScreenArr[0] = canvas.width;
    classifyScreenArr[1] = canvas.height;
    device.queue.writeBuffer(
      classifyShader.uniforms.screenParams.getGPUBuffer(device),
      0,
      classifyScreenArr,
    );
    decideScreenArr[0] = canvas.width;
    decideScreenArr[1] = canvas.height;
    device.queue.writeBuffer(
      decideShader.uniforms.screenParams.getGPUBuffer(device),
      0,
      decideScreenArr,
    );
    // Refine: canvas + the maxAdaptive bump-allocator budget (canvas-derived — legitimately live).
    refineScreenArr[0] = canvas.width;
    refineScreenArr[1] = canvas.height;
    refineScreenArr[2] = probeCounts.maxAdaptive;
    device.queue.writeBuffer(
      refineShader.uniforms.screenParams.getGPUBuffer(device),
      0,
      refineScreenArr,
    );
    // Args params: numUniform + maxAdaptive.
    argsParamsArr[0] = probeCounts.numUniform;
    argsParamsArr[1] = probeCounts.maxAdaptive;
    device.queue.writeBuffer(argsShader.uniforms.params.getGPUBuffer(device), 0, argsParamsArr);
    // Gather screenParams: canvas + frame index (mod 1024 for f32 exactness — the golden-angle
    // rotation + round-robin seed) + maxAdaptive (caps the adaptive counter). invViewProj = the
    // reverse-Z inverse-VP; prevViewProj = LAST frame's forward matrix (snapshotted below, AFTER
    // the uploads). Uploaded to BOTH gather shaders (uniform + adaptive) — they share the same
    // values, only IS_ADAPTIVE (baked) differs, and each has its own uniform buffers.
    screenParamsArr[0] = canvas.width;
    screenParamsArr[1] = canvas.height;
    screenParamsArr[2] = frameIndex % 1024;
    screenParamsArr[3] = probeCounts.maxAdaptive;
    for (const s of [gatherUniformShader, gatherAdaptiveShader]) {
      device.queue.writeBuffer(s.uniforms.screenParams.getGPUBuffer(device), 0, screenParamsArr);
      device.queue.writeBuffer(s.uniforms.invViewProj.getGPUBuffer(device), 0, screenInvArr);
      device.queue.writeBuffer(s.uniforms.prevViewProj.getGPUBuffer(device), 0, prevViewProjArr);
    }
    // Snapshot THIS frame's forward viewProj for next frame's reprojection (the only retained copy).
    prevViewProjArr.set(viewProjMatrix as Float32Array);
  }

  // Pass A0 — uniform placement. One thread per coarse 16px tile → probeData[tileIdx] (repr pixel,
  // level 0). Uploads the whole chain's uniforms first.
  function probeClassify(encoder: GPUCommandEncoder) {
    uploadProbeUniforms();
    const pass = encoder.beginComputePass();
    pass.setPipeline(classifyPipeline);
    pass.setBindGroup(0, classifyGroup0);
    pass.setBindGroup(1, classifyGroup1);
    pass.setBindGroup(2, classifyGroup2);
    pass.dispatchWorkgroups(
      Math.ceil(screenGrid.w / CLASSIFY_WG),
      Math.ceil(screenGrid.h / CLASSIFY_WG),
      1,
    );
    pass.end();
  }

  // Adaptive refinement, TWO encoder-barriered passes:
  //   1. DECIDE (one thread per coarse tile): raw light trigger (SH spread over the 3×3 uniform
  //      neighborhood) → the persistent per-tile hysteresis counter. lightThresh large ⇒ never
  //      triggers ⇒ counters decay to 0 ⇒ the flat uniform atlas (the uniform-only A/B).
  //   2. REFINE (one thread per fine cell): tiles whose counter is ACTIVE spawn their FULL fine
  //      lattice (cell center + the classify-style stable jitter); the resolve then uses the fine
  //      lattice INSTEAD of the tile's uniform probe (tileHeader > 0 → dense replaces sparse).
  function probeRefine(encoder: GPUCommandEncoder) {
    const decide = encoder.beginComputePass();
    decide.setPipeline(decidePipeline);
    decide.setBindGroup(0, decideGroup0[curSet]); // inSh* = THIS frame's write set (gatherUniform ran)
    decide.setBindGroup(1, decideGroup1);
    decide.setBindGroup(2, decideGroup2);
    decide.dispatchWorkgroups(
      Math.ceil(screenGrid.w / DECIDE_WG),
      Math.ceil(screenGrid.h / DECIDE_WG),
      1,
    );
    decide.end();

    const cellPx = Math.max(1, Math.floor(probeTile(config) / probeRefineDiv(config)));
    const cellsX = Math.ceil(canvas.width / cellPx);
    const cellsY = Math.ceil(canvas.height / cellPx);
    const pass = encoder.beginComputePass();
    pass.setPipeline(refinePipeline);
    pass.setBindGroup(0, refineGroup0);
    pass.setBindGroup(1, refineGroup1);
    pass.setBindGroup(2, refineGroup2);
    pass.dispatchWorkgroups(Math.ceil(cellsX / REFINE_WG), Math.ceil(cellsY / REFINE_WG), 1);
    pass.end();
  }

  // Pass B — build the indirect gather args from the (capped) adaptive count. Single thread.
  function probeBuildArgs(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(argsPipeline);
    pass.setBindGroup(0, argsGroup0);
    pass.setBindGroup(1, argsGroup1);
    pass.setBindGroup(2, argsGroup2);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
  }

  // Gather UNIFORM (DIRECT dispatch, runs BEFORE refine). ceil(numUniform / GATHER_WG) workgroups
  // over the uniform block → raw SH into atlas rows [0, gh). Refine reads this SH for its radiometric
  // (light-adaptive) subdivision trigger. numUniform is CPU-known, so no indirect args needed.
  function gatherUniform(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(gatherUniformPipeline);
    pass.setBindGroup(0, gatherUniformGroup0[curSet]); // history = spTex[1 - curSet] sampled views
    pass.setBindGroup(1, gatherUniformGroup1);
    pass.setBindGroup(2, gatherUniformGroup2[curSet]); // storage-writes spTex[curSet]
    pass.dispatchWorkgroups(Math.ceil(probeCounts.numUniform / GATHER_WORKGROUP), 1, 1);
    pass.end();
  }

  // Gather ADAPTIVE (INDIRECT dispatch, runs AFTER buildArgs). dispatchWorkgroupsIndirect launches
  // over exactly the adaptiveCount probes refine placed (PASS B wrote the args) → raw SH into atlas
  // rows [gh, ...). With no adaptive probes the counter is 0 → args = [0,1,1] → 0 groups → no-op.
  function gatherAdaptive(encoder: GPUCommandEncoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(gatherAdaptivePipeline);
    pass.setBindGroup(0, gatherAdaptiveGroup0[curSet]); // history = spTex[1 - curSet] sampled views
    pass.setBindGroup(1, gatherAdaptiveGroup1);
    pass.setBindGroup(2, gatherAdaptiveGroup2[curSet]); // storage-writes spTex[curSet]
    pass.dispatchWorkgroupsIndirect(probeBufs.args, 0);
    pass.end();
  }

  // Throttled async budget readback (~every 30 calls). Copies the counter to a staging buffer in a
  // standalone encoder, maps it, and surfaces the live adaptive count + a sticky BUDGET-EXCEEDED
  // flag (raised by the refine pass on an overflowing atomicAdd) for the GUI + a console.warn.
  let adaptiveProbeCount = 0;
  let budgetExceeded = false;
  let readbackInFlight = false;
  let readbackFrame = 0;
  const counterStaging = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  async function pollBudget() {
    readbackFrame++;
    if (readbackInFlight || readbackFrame % 30 !== 0) {
      return;
    }
    readbackInFlight = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(probeBufs.counter, 0, counterStaging, 0, 8);
    device.queue.submit([enc.finish()]);
    try {
      await counterStaging.mapAsync(GPUMapMode.READ);
      const a = new Uint32Array(counterStaging.getMappedRange().slice(0));
      counterStaging.unmap();
      adaptiveProbeCount = Math.min(a[0], probeCounts.maxAdaptive);
      budgetExceeded = a[1] !== 0;
      if (budgetExceeded) {
        console.warn(
          `[screen-probe] adaptive budget ${probeCounts.maxAdaptive} exceeded (allocated ${a[0]}); raise adaptiveFraction or tile`,
        );
      }
    } finally {
      readbackInFlight = false;
    }
  }

  // Recreate the canvas-derived screen-probe atlas + indirection buffers from the current grid +
  // adaptiveFraction. Called on resize / tile / adaptiveFraction change; the callers rebuild the
  // bind groups after (buildScreenProbeGroups + buildConeGroup) since those reference these objects.
  function recreateScreenProbeResources() {
    screenGrid = screenProbeGridDims(canvas.width, canvas.height, probeTile(config));
    probeCounts = screenProbeCounts(screenGrid, adaptiveFraction, probeRefineDiv(config));
    // BOTH ping-pong sets are destroyed + recreated together (the old texel↔probe mapping is
    // meaningless at the new dims). New textures are zero-filled → zero history normal → the
    // gather's validity test rejects the history, so the first post-recreate frame is fresh-only
    // with no explicit invalidation flag.
    destroyScreenProbeTextures(spTex[0]);
    destroyScreenProbeTextures(spTex[1]);
    spTex = [
      createScreenProbeTextures(device, screenGrid, probeCounts.adaptiveRows),
      createScreenProbeTextures(device, screenGrid, probeCounts.adaptiveRows),
    ];
    destroyScreenProbeBuffers(probeBufs);
    probeBufs = createScreenProbeBuffers(device, probeCounts);
  }

  // Screen-probe DEBUG view: false-color the probe distribution into compositeOutput (same target as
  // composite → present() picks it up unchanged). Run INSTEAD of composite() when debugProbes is on.
  // Reads the atlas + indirection (via debugGroup0/1), NOT the lit scene — MUST run after gatherAdaptive.
  function probeDebug(encoder: GPUCommandEncoder) {
    debugParamsArr[0] = canvas.width;
    debugParamsArr[1] = canvas.height;
    device.queue.writeBuffer(debugShader.uniforms.params.getGPUBuffer(device), 0, debugParamsArr);
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: getDebugTargetView(), clearValue: [0, 0, 0, 1], loadOp: "clear", storeOp: "store" },
      ],
    });
    pass.setPipeline(debugPipeline);
    pass.setBindGroup(0, debugGroup0[curSet]); // pix of THIS frame's write set
    pass.setBindGroup(1, debugGroup1);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  // Grid rebuilt (buildGrid): rebuild the gather group0 from the fresh voxelRadiance + aniso volumes
  // (+ the classify/refine/args/debug groups + re-upload the gather grid uniforms).
  function rebindGrid() {
    buildScreenProbeGroups();
  }

  // emitterLights' onBuffersRecreated (lightsBuf grow): rebuild the gather group1 (binds lightsBuf).
  function rebuildGroups() {
    buildScreenProbeGroups();
  }

  // Canvas resized: recreate the canvas-derived atlas + indirection buffers, then rebuild the groups.
  function resize() {
    recreateScreenProbeResources();
    buildScreenProbeGroups();
  }

  // Explicit rebuild (config change): recompile every BAKED shader of the chain with the CURRENT
  // config — the two gathers (hysteresis / bilateral weights / aniso mode / aimed group) AND the
  // classify/decide/refine/debug passes (tile / divisor / ring thresholds; decide also the light
  // trigger). The tile + refine divisor ALSO size the atlas/buffers, so the resources are
  // recreated from the same config, the groups rebuilt, and onResourcesRecreated fired (the cone
  // resolve rebinds the recreated buffers). Only args (pure counter math) survives untouched.
  function rebuild(cfg: VoxelBakedConfig) {
    config = cfg;
    gatherUniformShader.destroy();
    gatherAdaptiveShader.destroy();
    gatherUniformShader = new GPUShader(createScreenProbeShaderMeta(cfg, false));
    gatherAdaptiveShader = new GPUShader(createScreenProbeShaderMeta(cfg, true));
    gatherUniformPipeline = gatherUniformShader.getComputePipeline(device, "main");
    gatherAdaptivePipeline = gatherAdaptiveShader.getComputePipeline(device, "main");
    classifyShader.destroy();
    classifyShader = new GPUShader(createClassifyShaderMeta(cfg));
    classifyPipeline = classifyShader.getComputePipeline(device, "main");
    decideShader.destroy();
    decideShader = new GPUShader(createDecideShaderMeta(cfg));
    decidePipeline = decideShader.getComputePipeline(device, "main");
    refineShader.destroy();
    refineShader = new GPUShader(createRefineShaderMeta(cfg));
    refinePipeline = refineShader.getComputePipeline(device, "main");
    debugShader.destroy();
    debugShader = new GPUShader(createDebugShaderMeta(cfg));
    debugPipeline = debugShader.getRenderPipeline(device, "vs_main", "fs_main", {
      targetFormat: "rgba16float",
      withBlending: false,
    });
    // Tile / refine divisor may have changed → recreate the resources they size, from the SAME
    // config the shaders just baked (the strides cannot drift). NOTE: a fine divisor can exceed
    // SCREEN_PROBE_K adaptive probes/tile — the surplus is dropped from the per-tile list.
    recreateScreenProbeResources();
    buildScreenProbeGroups();
    onResourcesRecreated();
  }

  // Adaptive budget fraction (maxAdaptive = numUniform × fraction). Runtime state (no shader reads
  // it — only the buffer sizing). Resizes the atlas + all indirection buffers.
  function setAdaptiveFraction(fraction: number) {
    adaptiveFraction = Math.max(0, fraction);
    recreateScreenProbeResources();
    buildScreenProbeGroups();
    onResourcesRecreated();
  }

  // Toggle the probe-distribution debug view (probeDebug replaces composite) — live, no rebuild.
  // (The tuning knobs that used to have live setters here — lightThresh, normalPow/planeK,
  // resolveRadius, temporal hysteresis, foveated rings — are BAKED from VoxelBakedConfig now:
  // mutate the config and rebuild().)
  function setDebugProbes(on: boolean) {
    debugProbes = on;
  }

  // Camera-following grid: re-upload ONLY uGridOrigin to both gather shaders (originArr refreshed
  // by the caller — carries the zoom-ladder cellSize in .w too). A uniform write — dims/textures
  // unchanged, no bind-group rebuild.
  function uploadGridOrigin() {
    for (const s of [gatherUniformShader, gatherAdaptiveShader]) {
      device.queue.writeBuffer(s.uniforms.gridOrigin.getGPUBuffer(device), 0, originArr);
    }
  }

  return {
    // Pass methods.
    probeClear,
    probeClassify,
    probeRefine,
    probeBuildArgs,
    gatherUniform,
    gatherAdaptive,
    probeDebug,
    pollBudget,
    // Lifecycle.
    rebindGrid,
    rebuildGroups,
    resize,
    rebuild,
    uploadGridOrigin,
    // For the CONE cluster (still in createVoxelSystem, consumes these).
    getSpTex: () => spTex,
    getCurSet: () => curSet,
    getProbeBufs: () => probeBufs,
    // Setters (the one non-config sizing knob + the debug toggle — everything else is baked config).
    setAdaptiveFraction,
    setDebugProbes,
    // Getters.
    get adaptiveFraction() {
      return adaptiveFraction;
    },
    get adaptiveProbeCount() {
      return adaptiveProbeCount;
    },
    get budgetExceeded() {
      return budgetExceeded;
    },
    get debugProbes() {
      return debugProbes;
    },
  };
}
