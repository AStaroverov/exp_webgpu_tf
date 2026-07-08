# Probe-Centric GI Migration — Implementation Plan

> Implementation document. Goal: migrate the VCT lighting stack from **per-pixel cone tracing**
> (aimed emitter cones + AO marched in `voxelCone.shader.ts` at half res) to a **probe-centric
> final gather** (Lumen-style): ALL cone tracing lives on the screen probes and is temporally
> amortized; the per-pixel pass keeps only the cheap SH resolve, short screen-space contact
> shadows, and AO. Motivation: per-pixel × per-light × per-step is the fundamental cost — profiling
> showed no single block dominates (every knob buys 1–3 ms), so the win must come from deleting a
> multiplier, not tuning one. This also unlocks the "hundreds of emitters at constant cost" goal:
> per-pixel aimed cones are O(lights) per pixel and will never scale there.
>
> Companion docs: `voxel-cone-tracing-impl.md` (the current VCT design),
> `cone-fullres-direct-split.md`.

---

## Architecture: before / after

```
BEFORE:
  probes (fill cones only, 16/probe, fresh every frame)
      └► cone pass @ half res, PER PIXEL:
             aimed cone × up to 8 emitters × 32 steps × up to 4 fetches   ← dominant cost
             + resolve_screen_probes (SH fill)
             + 2 AO cones
      └► composite (upsample + apply)

AFTER:
  probes (fill cones + aimed emitter cones + aniso far-field, TEMPORALLY accumulated)
      └► cone pass @ half res, PER PIXEL:
             resolve_screen_probes (SH fill — now carries emitter light too)
             + screen-space contact shadows (short depth-buffer march)
             + 2 AO cones (iso pyramid only)
      └► composite (unchanged)
```

Trace work moves from `(W/2 × H/2) pixels × lights` to `~(W/16 × H/16) probes × lights`
(≈ 50–60× fewer traced points), then is cut a further 2–4× by temporal amortization.

**The one quality decision this bakes in:** emitter shadows become probe-resolution + contact
shadows (the Lumen model), not per-pixel cone-sharp. Stage 1 keeps a config switch so the old
path stays comparable until the new one is accepted; the switch is deleted in Stage 4.

---

## Stage 0 — Baseline capture (no behavior change)

**Goal:** numbers to compare every later stage against.

**Work:**

- Ensure per-pass GPU timings are readable (timestamp queries or, failing that, the existing
  A/B-toggle methodology) for: screen-probe gather (uniform + adaptive), cone pass, composite.
- Record on the perf scene: total frame, per-pass ms, and 4 screenshots — (S1) emitter near a
  wall casting a shadow, (S2) two bright emitters on opposite sides of one object, (S3) object
  in contact with the floor near an emitter (contact shadow), (S4) camera-motion clip for
  later ghosting checks.

**Live verification:** none (no change). Deliverable = a short table + screenshots committed or
kept alongside this doc. Every later stage re-measures the same scene.

---

## Stage 1 — Emitter light moves to the probes; aimed cones leave the pixel pass ✅ DONE

> **Status: DONE (and the probe-centric path is ACCEPTED; the switch was deleted in Stage 4).**
> One deliberate deviation from the plan below: the SH projection weight of an aimed cone is
> **`W_AIMED = 4π/3`**, NOT the light's solid angle `Ω ≈ π·(r/d)²`. The analytic direct term is
> already an integrated, irradiance-scale quantity (`atten` carries the distance falloff), so an
> Ω energy weight would apply the falloff twice; Ω only sets the trace APERTURE (penumbra width).
> 4π/3 is the weight for which `sh_avg_radiance` reconstructs ≈ `ndl·L` at `ndl = 1`, i.e. the
> old per-pixel scale (within SH-L1's softening). It is independent of `conesPerProbe`, as
> required.

**Goal:** the dominant per-pixel cost (aimed cones) is deleted; emitters still light the scene
via the probe SH field. Biggest perf win of the whole migration; gated behind a switch.

### 1a. Probe gather traces the emitters (`voxelScreenProbe.shader.ts`)

- Add to group 0 the same uniforms the cone shader has today: `lights: array<vec4<f32>, 8>`
  (center + radius), `lightColor: array<vec4<f32>, 8>` (rgb + intensity), and a live light count
  lane. Uniforms only — the 4-storage-texture cap on group 2 is untouched.
- After the fill-cone loop in `main`, add the aimed loop ported from `voxelCone.shader.ts`
  (the `(a) AIMED cones` block): per emitter — direction/distance, distance falloff `atten`
  (EMITTER_FALLOFF), the cheap contribution-cull (`< 0.003` before tracing), angular-size
  aperture, `trace` to the light, analytic direct + bleed-cancel ("white shadow" fix).
- Project the result into the same `cR/cG/cB` SH-L1 accumulators along the emitter direction.
  **Correctness trap:** the old per-pixel formula multiplies by `ndl`; SH storage holds
  *radiance* and the cosine is applied at resolve time by `sh_avg_radiance` — drop the `ndl`
  factor when porting or the cosine is applied twice (emitters come out too dim at grazing
  angles). Keep only the `ndl <= 0` hemisphere rejection.
- Weight: an aimed cone is a delta-ish direction, not a `dw = 2π/C` fill cone. Project as
  `L_emitter * Y(dir)` with a solid-angle weight derived from the light's angular size
  (`Ω ≈ π·(r/d)²`, clamped), NOT with the fill cones' `dw` — otherwise emitter energy scales
  with `conesPerProbe`.
- Move the 6 aniso volume bindings + `sample_aniso`/`sample_radiance` from the cone shader
  INTO this shader: the far-field anti-leak matters exactly where long cones are traced, and
  after this stage that is here. `trace_probe_cone` gains the `dir`-aware radiance fetch.

### 1b. Cone shader slims down (`voxelCone.shader.ts`)

- Delete: the aimed block, `lights`/`lightColor` uniforms, `EMITTER_*` consts, the 6 aniso
  bindings + `sample_aniso` (AO cones read the iso pyramid only — `sample_radiance` collapses
  to the plain `textureSampleLevel`), `directEmitters`.
- Output becomes `fillAvg * GI_STRENGTH` (+ AO in .a). `uParams2.z` (aniso toggle) now controls
  the probe gather instead.

### 1c. Wiring + config

- `createVoxelSystem.ts`: upload `coneLightsArr`/`coneLightColorsArr` to the probe shaders
  (both uniform + adaptive variants share the meta factory, so this is one code path); rebuild
  the probe bind groups with the aniso views; shrink the cone bind group.
- `voxelConfig.ts`: `aimedSteps`, `aimedAlphaCut`, `emitterDirect`, `emitterFalloff` move to the
  probe-pass section (they now bake into the probe shader). Add
  `emitterConesOnProbes: boolean` — `false` = old per-pixel path (kept fully working),
  `true` = new path. Both paths compile; the flag picks which shader gets the light uniforms
  and which shader contains the aimed loop.

### Live verification (perf scene, flip `emitterConesOnProbes` back and forth)

| Check | Expect |
| --- | --- |
| Cone-pass GPU time | drops by the full former aimed cost (the biggest single delta of the migration); probe gather grows only slightly (~2k probes × lights ≪ former pixel count × lights) |
| S1 (emitter near wall) | light + a *soft, low-frequency* shadow survives. Softness at probe resolution is **expected and accepted here** — crispness returns in Stage 2 |
| S2 (two opposing emitters) | both light the object; check for SH-L1 "mush" (two opposing lobes cancel in L1). Some loss is a known SH-L1 limit — record it; if unacceptable, the fallback is per-probe octahedral storage (out of scope, noted as the escape hatch) |
| Brightness parity | overall emitter energy within ~±20% of the old path (fix the Ω weight if it scales with `conesPerProbe`) |
| Lights = 0 | new path == old path pixel-identical (aimed loop is a no-op both ways) |
| Adaptive probes | subdivision still reacts to emitter light (refine reads the raw SH, which now includes emitters — it should get *more* responsive, not less) |

**Rollback:** `emitterConesOnProbes = false`.

---## Stage 2 — Screen-space contact shadows (quality, not perf) ⛔ SKIPPED

> **Status: SKIPPED by decision.** Emitter shadows will be redesigned from scratch later — the
> screen-space contact-shadow march below was prototyped and then removed in Stage 4 (no
> contact-shadow code remains in the tree). Until the redesign, emitter shadows are
> probe-resolution soft (the accepted Stage-1 quality level). The plan below is kept for
> reference only.

**Goal:** restore high-frequency occlusion near contacts that Stage 1 gave up. Cheap per-pixel
work in the cone pass (it already has depth + reconstructed P).

**Work (in `voxelCone.shader.ts`, after the probe resolve):**

- A short screen-space ray march against the reverse-Z depth buffer: 8–16 steps, world reach
  ~0.5–1.0 (config), thickness tolerance ~1–2 voxel cells, dithered start (reuse `jrad`).
  Hit ⇒ occlusion factor for that direction.
- Direction policy, staged:
  1. **Now (≤8 lights):** march toward each live emitter, gated by the same cheap
     contribution-cull the aimed cones used — most pixels march toward 0–2 lights.
  2. **Later (hundreds of lights):** ONE march along the resolved SH dominant direction
     (`normalize(sum L1.xyz)`) — constant cost per pixel regardless of light count. Ship (1)
     first; (2) is a drop-in replacement of the direction loop.
- Apply as a multiplier on the probe fill: `fill *= 1 - k * occ` with `k` (config
  `contactShadowStrength`) so it darkens contacts without crushing the whole fill. New config
  knobs: `contactShadowSteps`, `contactShadowReach`, `contactShadowStrength` (0 disables ⇒
  byte-identical to Stage 1 output).

### Live verification

| Check | Expect |
| --- | --- |
| S3 (object on floor near emitter) | a readable dark contact edge reappears under/behind the object; compare against the Stage 0 screenshot — should approach the old aimed-cone crispness at contacts |
| Cone-pass GPU time | grows < ~0.5 ms |
| Off-screen occluder | its contact shadow disappears (screen-space limitation) — verify the remaining probe (voxel) shadow still provides the soft term so it degrades gracefully, not to zero |
| `contactShadowStrength = 0` | identical to Stage 1 |

**Rollback:** `contactShadowStrength = 0`.

---

## Stage 3 — Temporal accumulation on the probe atlas ✅ DONE

> **Status: DONE.** Ping-pong atlas history, reprojection through the previous frame's forward
> viewProj, validation via the SHARED `sp_plane_normal_weight`, per-frame golden-angle rotation of
> the Fibonacci cone set (gated on usable history), `hysteresis = 0` = byte-identical fresh-only
> path.

**Goal:** cut the (now dominant) probe-gather budget 2–4× by amortizing across frames. Only safe
after Stage 1 is accepted (amortizing the old per-pixel path is pointless).

### 3a. History resources (`voxelResources.ts`)

- Second (ping-pong) set of `shR/shG/shB` + second `pos/nrm` (the current-frame anchors already
  exist — history needs its own copy for validation). Swap roles each frame; recreate on resize
  like the rest of `ScreenProbeTextures`.

### 3b. Reprojection + blend

- In the probe gather (or a small compute right after it): reproject the probe's fresh anchor
  `P` with the **previous** frame's `viewProj` (new uniform; CPU keeps last frame's matrix) →
  previous screen pixel → previous uniform tile → history atlas texel.
- Validate with the SHARED `sp_plane_normal_weight` from `voxelProbeShared.wgsl.ts` (do not
  write a second bilateral): fresh `P/N` vs history `pos/nrm`. Weight ≈ 0 ⇒ disocclusion ⇒
  hysteresis 0 (use fresh only).
- Blend `sh = mix(fresh, history, hysteresis)`, `hysteresis` config ~0.85–0.9, `0` disables
  (⇒ identical to Stage 2 output). Adaptive probes: reproject against the history UNIFORM tile
  they fall in (adaptive slots are not stable frame-to-frame; the uniform layer is the
  persistent backbone).

### 3c. Spend the history: cut the fresh budget

- `conesPerProbe: 16 → 4–8` with a per-frame golden-angle rotation of the Fibonacci set
  (frame-index uniform lane) so successive frames fill different directions; temporal blend
  integrates them back to effective 16–32.
- Many emitters (when they arrive): round-robin — each frame traces lights `frame % K` per
  probe; history smooths.
- `adaptiveFraction 1.0 → ~0.5`: the comment in `voxelResources.ts` (`screenProbeCounts`)
  explicitly ties 1.0 to "we amortize NOTHING" — that premise is gone now.

### Live verification

| Check | Expect |
| --- | --- |
| Static scene, `conesPerProbe = 4`, hysteresis 0.9 | converged image ≈ Stage 2 at 16 cones (allow a few frames to converge); probe-gather GPU time ~⅓–¼ |
| S4 camera clip | GI lags ≤ 3–4 frames; NO smearing/ghosting on disocclusions (sharp camera turn, object driving past a wall) — failures point at the plane/normal validation, not the blend |
| Moving emitter | light pool follows with the same ≤ 3–4 frame lag; no stale bright trail |
| `hysteresis = 0` | identical to Stage 2 (fresh-only path intact) |
| Resize / grid rebuild | no crash, no one-frame garbage (history invalidated or aged out) |

**Rollback:** `hysteresis = 0` + `conesPerProbe = 16`.

---

## Stage 4 — Cleanup and consolidation ✅ DONE

**Goal:** delete the scaffolding; bank the structural simplifications.

**Done:**

- Deleted the old per-pixel aimed path + the `emitterConesOnProbes` switch: the probe-centric
  path is the ONLY path now. The cone shader keeps no emitter knowledge at all (no
  lights/lightColor uniforms, no aniso bindings, no `sample_aniso`, no `EMITTER_*`/`AIMED_*`
  consts) — it is a pure probe RESOLVE + 2 short AO cones on the iso pyramid. The aimed loop,
  aniso volumes and `sample_radiance` are unconditional in the probe gather.
- The Stage-2 contact-shadow prototype was also removed (Stage 2 skipped — see above).
- Updated `voxel-cone-tracing-impl.md` + the shader header comments to describe the
  probe-centric architecture.

**Deferred follow-ups (both require live GPU measurements — NOT implemented here):**

1. **Cone→composite pass merge.** Evaluate merging the slimmed cone pass into the composite
   (it is now resolve + 2 AO cones — the half-res HDR round-trip may cost more than it saves).
   Measure on the perf scene, then decide.
2. **AO cones → depth-based horizon AO (GTAO-style).** Replacing the 2 AO cones would kill the
   last per-pixel 3D fetches; keep only if it wins on the perf scene — the cones are ~1 ms.

**Live verification:** final side-by-side vs the Stage 0 baseline — S1–S4 screenshots + the
timing table. Target: cone pass ≈ resolve-only (~1 ms), probe gather = a fixed budget that no
longer scales with resolution or light count; total GI cost down by the aimed-block cost plus
2–4× on the gather.

---

## Stage dependencies & risk summary

| Stage | Depends on | Main win | Main risk |
| --- | --- | --- | --- |
| 0 | — | comparability | — |
| 1 | 0 | deletes the dominant per-pixel × per-light cost | soft emitter shadows until Stage 2; SH-L1 two-opposing-lights mush |
| 2 | 1 | restores contact crispness | screen-space misses off-screen occluders |
| 3 | 1 (2 recommended) | 2–4× on the probe budget | ghosting / disocclusion lag; +3 history textures |
| 4 | 1–3 accepted | code deletion, possible pass merge | — |

Escape hatch if SH-L1 proves too lossy for emitters (Stage 1, check S2): per-probe octahedral
radiance tiles (Lumen's actual storage) instead of SH-L1 — a bigger change, deliberately out of
scope for this migration.
