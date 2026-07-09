# Lighting / Voxel GI refactor — progress

> Living document: check boxes as steps land. Full rationale and recon are in the sections below.
> Rule: one step = one file = a green `tsc --noEmit` before the next.

## Progress

### Phase A — Scheme (prerequisite)

- [x] **Step 0.** `README.md` — whole-system scheme (resources, pass pipeline, data flow,
      baked-vs-dynamic, UBO layouts, ping-pong, no attributes). ✅

### Phase B — Shaders (low risk)

- [x] **Step 1.** `voxelTrace.wgsl.ts` — shared `unproject` (matrix-parameterized) + `build_basis`.
      ✅ Note: cone-trace is NOT shared — screen-probe's trace needs aniso volumes the cone shader
      deliberately omits. (File lives at Lighting root for now; moves to `shaders/shared/` in Step 6.)
- [x] **Step 2.** `voxelScreenProbe.shader.ts` — use shared `unproject`/`build_basis`, drop local
      copies. ✅ (Deferred: splitting this 750-line WGSL into sub-modules — risky, tsc can't validate
      WGSL; revisit only with a running-app check.)
- [x] **Step 3.** `voxelCone.shader.ts` — use shared `unproject`/`build_basis`, drop local copies. ✅
- [x] **Step 4.** `voxelComposite.shader.ts` — use shared `unproject`. ✅ (`CompositeFrame` cleanup
      moved to Step 7.)
- [x] **Step 5.** `voxelAnisoVolume.shader.ts` — 6 copy-paste blocks collapsed into a JS generator;
      output verified byte-identical to the original via git diff. ✅
- [ ] **Step 6.** Move all shaders into `shaders/`, unify `*ShaderMeta` exports, update imports.
      **Resequenced → done at the end** (after Phase D). The generic `shaderMeta` aliases are used
      20+ times each in `createVoxelSystem`; Phase D rewrites every one of those imports when it
      extracts the pipeline/pass modules, so renaming now would be redone. Fold the rename + the
      physical `shaders/` move into a single final reorganization once the module tree exists.

### Phase C — Uniform infrastructure

> **Resequenced → after Phase D.** These change UBO packing (`CF_*` offsets, shared grid/sun
> buffers). A wrong offset breaks rendering SILENTLY — tsc cannot catch it and WGSL isn't validated
> locally, so it needs a running-app check. Safest once the composite/uniform code is already
> isolated in its own module (Steps 11/18), so the change is small and reviewable.

- [ ] **Step 7.** `uniforms/frameBlock.ts` — named UBO-block helper; replace `CF_*` in composite.
- [ ] **Step 8.** Shared grid/sun UBOs; named lanes for `params2/params3/temporalParams`.

### Phase D — Split the monolith into PURE FUNCTIONS (no god-context)

> **Approach change (per user):** no big mutable `VoxelCtx` threaded everywhere. Instead extract
> each self-contained piece as a **pure function** with a narrow, explicit parameter list
> (factory-with-scratch when per-frame zero-alloc matters). `createVoxelSystem` keeps owning the
> GPU state and calls these. All tsc-green, logic byte-preserved. `createVoxelSystem.ts`: 2320 → 2032.

- [x] **Step D1.** `voxelizeCpu.ts` — pure CPU AABB/prefix-sum builder + `footprintHalfZ`. ✅
- [x] **Step D2.** `lightClustering.ts` — pure CPU clustered light culling. ✅
- [x] **Step D3.** `sunViewProj.ts` — pure sun ortho view-proj fit (factory-with-scratch, 0 alloc). ✅
- [x] **Step D4a.** `passes/mipPass.ts` — `runMips` / `runAnisoBase` / `runAnisoMips` (the cleanest
      pass group: read-only over state). The caller keeps thin wrappers that build a scoped deps bundle
      from live locals — no god-context. Pattern established here for the rest. ✅
- [ ] **Step D4b.** Remaining passes (`sunDepth`, `voxelize`, probe chain + `uploadProbeUniforms`,
      `cone`, `composite`/`probeDebug`) — same `passes/` + scoped-bundle pattern, but heavier: large deps
      bundles and several WRITE shared state (`scatterTotal`, `curSet`/`frameIndex`, `sunWorldTexel`).
      Higher churn / lower benefit; best done with an in-app run per pass. Deferred pending that loop.

#### Remaining in `createVoxelSystem` (deliberately NOT extracted, with reasons)

The pure, cleanly-separable logic is now out. What's left is intrinsically GPU-stateful wiring where
the "pure function" model doesn't fit and extraction would need either wide param threading or a
shared state bundle (which we're avoiding):

- **Per-pass dispatch fns** (`mips`/`anisoBase`/`anisoMips`/`sunDepth`/`voxelize`/probe chain/`cone`/
  `composite`/`probeDebug`) — thin GPU glue (~15–30 lines each) that issue commands over bind groups
  reassigned by `buildGrid`/`recreate`. Impure; read best co-located with the state they drive.
- **Bind-group builders + `buildGrid`** — create ~35 GPU bind groups and reassign ~30 state fields;
  deeply coupled to the handle set.
- **Pipeline/sampler setup** — one-shot creation; `rebuild()` reassigns 4 shaders/pipelines.
- **`uploadProbeUniforms` / config setters** — pack scratch + upload; tied to the shader handles.

> If more file-splitting is wanted here, the honest options are (a) a small scoped state bundle passed
> to a `passes/` module, or (b) positional-arg pass functions. Both are structural-only (no behavior
> change) and need an in-app run to confirm. Flagged for a decision rather than forced.

### Phase E — Orchestration dedup

- [x] **Step 23.** `voxel.renderFrame(encoder)` — the full canonical GI scenario (sunDepth-if-sun →
      voxelize → mips → aniso base/mips → probe chain → cone → composite/debug) lives on the system.
      demo's non-PERF branch AND engine's render loop both replaced with a single `voxel.renderFrame`
      call (frameTick before, present after). engine's historical no-aniso chain unified to the full
      one (per user — the difference was only historical). demo's PERF branch keeps its per-pass
      toggles. ✅

### Phase F — Final cleanup

- [x] **Step 24.** Deduped the extra `voxelSampler` (screenProbe now takes the shared one); dead doc
      links (`docs/voxel-cone-tracing-impl.md`, `docs/probe-centric-gi-migration.md`) repointed to
      `./README.md`; `oxfmt` applied; tsc green (renderer + engine), Lighting oxlint clean. ✅

---

## Result

`createVoxelSystem.ts`: **2320 → 506 lines** — now a thin assembler (build the 8 sub-systems + grid
`buildGrid` conductor + `renderFrame` scenario + config API) with NO shader/pipeline/bind-group/pass
GPU details of its own. Sub-system modules: sunShadowSystem, mipPyramidSystem, anisoVolumeSystem,
compositeSystem, emitterLightsSystem, voxelizeSystem, screenProbeSystem, coneSystem (+ the pure
helpers voxelizeCpu, lightClustering, sunViewProj, voxelTrace, passes/mipPass). The frame reads as a
scenario via `voxel.renderFrame(encoder)`.

Grid (`voxelRadiance` + `buildGrid`) intentionally stays in `createVoxelSystem` as the conductor that
rebinds every sub-system on a grid rebuild (agreed — it IS the orchestrator, not a leaf cluster).

---

## Decisions (agreed)

- **Comments — touch minimally:** remove only obvious cruft (stale, duplicated, dead links to the
  nonexistent `docs/voxel-cone-tracing-impl.md`, `docs/probe-centric-gi-migration.md`); valuable
  rationale moves with its code or into the README.
- **Full split of `createVoxelSystem` via `VoxelCtx`, iterative** (verify after every step).
- **Scheme lives in `Lighting/README.md`.**
- **Pass-order dedup in `voxel.renderFrame(encoder)`.**

## Guardrails

- No behavior / public-API changes. External callers (`demo/index.ts`,
  `engine/createRenderTarget.ts`, `RenderDI.VoxelSystem`) stay intact; only import paths of moved
  files change.
- Pass order and inter-compute-pass barriers are load-bearing (gatherUniform BEFORE refine;
  clear/scatter as separate passes; emitter-scatter last). Do not reorder.
- Ping-pong parity, "no per-frame createBindGroup", uncapped lights — preserved.

## Verification (after each step)

1. `npx tsc --noEmit -p packages/renderer/tsconfig.json` — green.
2. `npm --prefix packages/renderer run dev` — GI/shadows/screen-probe/debug identical, console clean.
3. Final: `npm --prefix packages/renderer run build` + `npm run lint`.
