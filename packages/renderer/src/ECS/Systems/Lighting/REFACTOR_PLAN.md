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

### Phase D — Split the monolith via VoxelCtx (iterative)
> Progress so far (all tsc-green, logic byte-preserved): the two fattest PURE-CPU chunks are out —
> `voxelizeCpu.ts` (`buildVoxelAABBs` + `footprintHalfZ`, Step 14's CPU core) and
> `lightClustering.ts` (`assignLightClusters`, Step 19's core). `createVoxelSystem.ts`: 2320 → 2141.
> These are low-risk (narrow interfaces, no shared GPU state). The remaining GPU-coupled extraction
> (pipelines/bindgroups/passes/VoxelCtx threading) needs an **in-app run to confirm** — tsc does not
> validate WGSL or per-frame GPU state, so it is the natural point to smoke-test `npm run dev` before
> the heavier surgery.
- [x] **Step 14a.** `voxelizeCpu.ts` — pure CPU AABB/prefix-sum builder + `footprintHalfZ`. ✅
- [x] **Step 19a.** `lightClustering.ts` — pure CPU clustered light culling. ✅
- [ ] **Step 9.** `voxelContext.ts` — `VoxelCtx` type + init.
- [ ] **Step 10.** `pipelines/voxelPipelines.ts` — 13 `GPUShader` + sampler + empty group-1.
- [ ] **Step 11.** `uniforms/voxelUniforms.ts` — scratch + `uploadProbeUniforms` + `buildSunViewProj`.
- [ ] **Step 12.** `bindgroups/voxelBindGroups.ts` — all `buildXGroup` + `buildGrid`.
- [ ] **Step 13.** `passes/sunShadowPass.ts` — `sunDepth`.
- [ ] **Step 14.** `passes/voxelizePass.ts` — `voxelize` + CPU AABB/prefix-sum + `footprintHalfZ`.
- [ ] **Step 15.** `passes/mipPass.ts` — `mips` / `anisoBase` / `anisoMips`.
- [ ] **Step 16.** `passes/probePass.ts` — the whole probe chain.
- [ ] **Step 17.** `passes/conePass.ts` — `cone`.
- [ ] **Step 18.** `passes/compositePass.ts` — `composite` / `probeDebug`.
- [ ] **Step 19.** `lights/voxelLights.ts` — `setLights` + clustering + `createLightEmitterSystem`.
- [ ] **Step 20.** `budget/voxelBudget.ts` — `pollBudget` + readback + `recreateScreenProbeResources`.
- [ ] **Step 21.** `api/voxelConfigApi.ts` — `rebuild` / `recreate` / `setXxx` / getters.
- [ ] **Step 22.** `createVoxelSystem.ts` — shrink to a thin assembler.

### Phase E — Orchestration dedup
- [ ] **Step 23.** `renderFrame.ts` — `voxel.renderFrame(encoder)`; replace the copies in demo + engine.

### Phase F — Final cleanup
- [ ] **Step 24.** Minimal comment cleanup (stale/dupes/dead links), `fmt` + `lint`.

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
