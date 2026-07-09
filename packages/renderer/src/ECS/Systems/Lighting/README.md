# Lighting System — Voxel Global Illumination (VCT + Screen Probes)

Architecture of the renderer's lighting system: a voxel indirect-lighting pipeline (Voxel Cone
Tracing) with screen-space probes (SH-L1) and a direct directional sun with a shadow map. It all
runs on WebGPU on top of the in-repo shader framework (`ShaderMeta` / `GPUShader` / `GPUVariable`).

---

## 1. Lighting model

Two light sources add up in the final composite:

- **Directional sun (`SunLight`)** — a DIRECT term (N·L) with a crisp cast shadow from a sun-POV
  depth map (`sunDepth` pass). The sun is also injected (shadowed) into the voxel volume by
  `voxelize`, so it contributes a GI bounce as well. Dormant when the sun is disabled (`sun.w==0`).
- **Emitters (point lights, `LightEmitter` component)** — injected into the voxel volume and
  gathered by the cone GI (aimed cones toward emitters + fill hemisphere cones). This is the
  composite's "indirect" term. The emitter list is auto-discovered from the ECS each frame
  (`createLightEmitterSystem`) and is NOT capped by count — cost stays constant via round-robin:
  each probe traces `aimedPerFrame` cones from its own cluster cell.

---

## 2. GPU resources

Created in `voxelResources.ts`. All 3D textures are `rgba16float`, `STORAGE_BINDING |
TEXTURE_BINDING` (written by a compute pass via `textureStore`, read later as a sampled texture).

| Resource                          | Type / format                       | Writer                                        | Reader                                  | Notes                                                                                                |
| --------------------------------- | ----------------------------------- | --------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------- | --- | --------------------- |
| `voxelRadiance`                   | 3D rgba16float + **mip pyramid**    | `voxelize` (mip0), `mips` (mip1..N)           | `anisoBase`, screen-probe gather        | rgb = direct-lit radiance (sun N·L·vis) + emission; a = occupancy. ~38 MB on the default grid        |
| `voxelEmission`                   | 3D rgba16float, single mip          | `voxelize` (emitter scatter)                   | `voxelize` (occluder merge + mip0 copy) | Emitter-class scatter target; merged into `voxelRadiance` mip0 (sum rgb, max a) — kills the emitter/occluder shared-voxel overwrite |
| aniso ×6 (`negX/posX/…/posZ`)     | 3D rgba16float ×6 + mips, **½ res** | `anisoBase` (lvl0), `anisoVolume` (lvl c→c+1) | screen-probe gather                     | Directional volumes (anti-leak); the cone picks the 3 volumes facing the cone dir and blends by dir² |
| screen-probe atlas: `shR/shG/shB` | 2D rgba16float                      | gather                                        | `refine`, cone resolve, debug           | SH-L1 coefficients (per channel)                                                                     |
| `nrm`                             | 2D rgba16float                      | gather                                        | gather (history validity), cone resolve | probe world normal;                                                                                  | nrm | ²≈0 = invalid history |
| `pix`                             | 2D rgba32float                      | gather                                        | cone resolve, debug                     | probe representative pixel + validity + footprint                                                    |
| `pos`                             | 2D rgba32float                      | gather                                        | cone resolve                            | probe world anchor P (32-bit for plane-reject precision)                                             |
| screen-probe buffers: `counter`   | atomic<u32>×2                       | refine (atomicAdd), clear                     | args                                    | [0]=bump allocator, [1]=sticky budget-exceeded                                                       |
| `data`                            | array<vec4<u32>>                    | classify, refine                              | gather, cone                            | per-probe record (repr pixel + level), index = global slot                                           |
| `header`                          | array<atomic<u32>>                  | refine, clear                                 | cone                                    | count of adaptive probes per coarse tile                                                             |
| `indices`                         | array<u32>, stride=K(=8)            | refine                                        | cone                                    | global slots of a tile's adaptive probes                                                             |
| `args`                            | array<u32,3>, INDIRECT              | args, clear                                   | `gatherAdaptive` (indirect dispatch)    | [ceil(total/WG),1,1]                                                                                 |
| `sunDepth`                        | 2D depth32float                     | `sunDepth`                                    | `voxelize`, `composite`                 | sun-POV shadow map                                                                                   |
| `coneOutput`                      | 2D HDR, **½ res**                   | `cone`                                        | `composite`                             | indirect + AO; composite bilinearly upsamples                                                        |
| `compositeOutput`                 | 2D HDR full-res                     | `composite` / `probeDebug`                    | present                                 | final image                                                                                          |
| `lightsBuf`                       | storage, grow-doubled               | `setLights`                                   | gather                                  | emitters (x,y,z,r, r,g,b,i) as 2×vec4                                                                |
| `clusterBuf`                      | storage                             | `setLights` (CPU clustering)                  | gather                                  | per-cell emitter lists                                                                               |

**Screen-probe atlas** is a flat probe atlas (not a literal screen grid): one probe per
`tile×tile` pixels (default 16, "Lumen DownsampleFactor"). The uniform block occupies rows
`[0,gh)` (identity tile→texel mapping); the adaptive block occupies rows `[gh, gh+adaptiveRows)`.
The atlas is canvas-derived → recreated on resize.

---

## 3. Pass pipeline (frame order)

The order is **load-bearing** (the encoder provides barriers between compute passes; do not
reorder). It lives in one place — `voxel.renderFrame(encoder)` — which both callers
(`demo/index.ts`, `engine/createRenderTarget.ts`) invoke (demo's PERF branch keeps per-pass toggles).

```
   [SDF G-buffer draw]  (outside this system: depth + normal + albedo + emission)
            │
            ▼
   sunDepth ─────────────► sunDepthTexture ──┐
            │                                 │ (sun shadow)
            ▼                                 │
   voxelize:                                  │
     clear voxelEmission                      │
     scatter emitters  (uPass=1) ─► voxelEmission
     copy voxelEmission ─► voxelRadiance mip0    (doubles as the mip0 clear)
     scatter occluders (uPass=0)  ◄───────────┘  injects shadowed sun; MERGES
            │  voxelRadiance mip0                voxelEmission (sum rgb, max a)
            ▼
   mips (mip1..N, one pass per level)
            │  voxelRadiance pyramid
            ▼
   anisoBase (lvl0 from iso mip0)  ──►  6 aniso volumes lvl0
            │
            ▼
   anisoMips (lvl c→c+1, per direction)  ──►  aniso pyramids
            │
            ▼
   screen-probe chain:
     probeClear (counter/header/args)
     probeClassify ─► data (uniform placement, 1 thread/tile)
     gatherUniform ─► SH atlas [0,gh)          (DIRECT dispatch; BEFORE refine!)
     probeRefine   ─► data/counter (adaptive spawn by SH radiometry)
     probeBuildArgs ─► args (indirect from counter)
     gatherAdaptive ─► SH atlas [gh,…)         (INDIRECT dispatch)
            │  SH atlas (+pos/nrm/pix), ping-pong curSet
            ▼
   cone (½-res): resolve screen probes (fill/bounce + emitters via SH) + AO cones
            │  coneOutput
            ▼
   composite (full-res): albedo·(ambient·AO + directSun·shadow + indirect) + emission,
                         ACES tonemap, sun PCF shadow, upsample coneOutput
            │
            ▼
   present(compositeOutput)      [or probeDebug instead of composite when voxel.debugProbes]
```

**Order subtlety:** `gatherUniform` runs BETWEEN `classify` and `refine` (not on the naive
chain), because `refine` reads its raw SH as the subdivision signal.

---

## 4. Screen probes: adaptivity and temporal accumulation

- **Adaptive density.** `refine` (16→8) atomically spawns extra probes where incoming light
  varies across the cage by more than `lightThresh`. A large `lightThresh` ⇒ no spawns ⇒
  counter=0 ⇒ gatherAdaptive is a no-op ⇒ exactly the flat atlas (uniform-only A/B). Budget
  `maxAdaptive` = `numUniform × adaptiveFraction`; overflow → sticky flag + `console.warn`
  (throttled readback `pollBudget`).
- **Temporal accumulation (ping-pong).** The system keeps TWO full atlas sets and swaps them by
  frame parity (`curSet = frameIndex & 1`). The current set is the gather's write target; the
  other is last frame's output = the history the gather reprojects (`prevViewProj`) and blends
  by `temporalHysteresis`. History validity = |nrm|²>0 (an invalid probe writes zero pos/nrm).

---

## 5. Configuration: baked vs dynamic

- **Baked** (`voxelConfig.ts`, `VoxelBakedConfig`, ~19 constants) — quality/tuning knobs that are
  CONSTANT during gameplay. Interpolated as WGSL `const`s in the `createConeShaderMeta` /
  `createCompositeShaderMeta` / `createScreenProbeShaderMeta` factories. Changing one → `voxel.rebuild()`
  (recompile affected shaders + rebuild their pipelines/bind groups).
- **Dynamic** (uniforms, per frame) — sun, camera `invViewProj`, emitter list, instance count,
  grid matrices, canvas size, temporal lanes, and the **grid origin**: the voxel box's XY origin
  follows the camera (`updateGridOrigin`, head of `setLights`/`renderFrame`), snapped to 4-voxel
  multiples so voxel centers in the overlap land on the same world points (no pan shimmer; iso
  mips 1–2 + aniso base keep their block partition). Toggle: `setFollowCamera` (GUI A/B).
- ⚠️ Minor boundary leak: some "baked" values also ride live uniforms (cone `params3` carries
  `tile`/normalPow/planeK/resolveRadius). Left as-is; documented here.

---

## 6. Uniforms and attributes

- **No vertex attributes.** All draws are attribute-less: fullscreen passes generate a 6-vertex
  triangle pair from a WGSL constant via `@builtin(vertex_index)` (`draw(6)`); voxelize/sunShadow
  draw an impostor cube `const CUBE: array<vec3,36>` as instances (`draw(36, instanceCount)`).
  Per-instance data flows through StorageRead buffers (`sceneInstances.*`), not attributes.
- **Uniforms.** Scalar/vec/mat UBOs are auto-created and auto-sized by the framework from the type
  (`getTypeTypedArray` + `getGPUBuffer`). Exceptions (hand-managed): the consolidated
  `CompositeFrame` UBO (manual `CF_*` offsets, `size:48/bufferSize:192`) and the screen-probe
  storage/indirect buffers (raw `createBuffer`, bound manually — the framework can't express
  INDIRECT). The `params2/params3/temporalParams` lane layouts are documented in comments at each site.

---

## 7. Key invariants (do not break)

- **No per-frame `createBindGroup`.** Bind groups are built at setup / `buildGrid` / `rebuild` and
  reused; ping-pong sets are switched by rebinding, not by rebuilding.
- **Barriers between compute passes** come from separate `beginComputePass/end`. `clear` and
  `scatter` are separate passes (else a race on `textureStore`). Occluders and emitters scatter
  into SEPARATE volumes (emitters → `voxelEmission`, then copy → mip0, then occluders merge on
  top) — neither class can overwrite the other's contribution on a shared voxel, and the write
  order stays deterministic (no flicker). Within one class, AABB overlap is last-writer-wins.
- **`gatherUniform` strictly BEFORE `refine`.**
- **Uncapped lights** + round-robin `aimedPerFrame` → cost constant in the light count.
- Ping-pong parity, mip chain one pass per level, aniso per direction.

---

## 8. File map

The directory is organized so the root shows only the entry point + docs, and the rest falls into
three concepts: **`core/`** (foundation shared by every stage), **`lights/`** (emitter input, prepared
outside `renderFrame`), and **`stages/`** (the GI pipeline — one numbered folder per stage, in frame
order). Each stage folder co-locates its sub-system (orchestration) + its shader(s) (GPU logic) + its
pure CPU helper (if any) — the three layers of one responsibility side by side.

```
Lighting/
  createVoxelSystem.ts              ← THE entry point: assembler + grid conductor + renderFrame scenario
  README.md                         ← this document

  core/                             ← foundation shared by every stage
    voxelConfig.ts                  ← baked config (WGSL consts, needs rebuild())
    voxelResources.ts               ← texture/buffer factories
    mipPass.ts                      ← shared dispatch primitive (runMips/runAnisoBase/runAnisoMips)
    shaders/
      voxelTrace.wgsl.ts            ← shared WGSL: unproject / build_basis
      voxelProbeShared.wgsl.ts      ← shared WGSL: probe pack / weight

  lights/                           ← INPUT: emitters (setLights, called outside renderFrame)
    emitterLightsSystem.ts          ← GPU light buffer + upload
    createLightEmitterSystem.ts     ← ECS discovery of LightEmitter entities
    lightClustering.ts              ← pure CPU clustered light culling

  stages/                           ← the GI pipeline, one folder per stage, in frame order
    1_sunShadow/                    ← sunShadowSystem + sunShadow.shader + sunViewProj (pure)
    2_voxelize/                     ← voxelizeSystem + voxelize.shader + voxelizeCpu (pure)
    3_mipPyramid/                   ← mipPyramidSystem + voxelMip.shader
    4_anisoVolume/                  ← anisoVolumeSystem + voxelAnisoBase/Volume.shader
    5_screenProbe/                  ← screenProbeSystem + voxelScreenProbe + voxelProbe{Classify,Refine,Args,Debug}.shader
    6_cone/                         ← coneSystem + voxelCone.shader
    7_composite/                    ← compositeSystem + voxelComposite.shader
```

`createVoxelSystem` builds each stage's `createXxxSystem(deps)` sub-system, owns the voxel grid, and
drives the frame via `renderFrame(encoder)` (the numbered stage order = the pass order in §3).

Public API (`ReturnType<typeof createVoxelSystem>`, see `RenderDI.VoxelSystem`): `renderFrame`, the
individual passes (`sunDepth/voxelize/mips/anisoBase/anisoMips/probe*/gather*/cone/composite/probeDebug`),
`setLights`, `pollBudget`, `rebuild`, `recreate`, the `setXxx` setters and getters, `coneOutputTexture` /
`compositeOutputTexture`.
