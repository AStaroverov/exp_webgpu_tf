# RC Phase 3 (Path A) — Screen-space RC + Temporal Accumulation (SVGF-style)

> **Goal:** kill the residual noise/blockiness of the screen-space Radiance Cascades
> **without changing the architecture**, by accumulating irradiance over frames
> (temporal reuse) — the same trick the surface-cache reference (`example/`) uses,
> adapted to our screen-space G-buffer pipeline. Spatial à-trous stays, but becomes a
> light cleanup pass instead of the sole denoiser.
>
> **Status:** design. Implement in two sub-steps (see Rollout).
>
> This is the **pragmatic** path (recommended near-term). The world-space surface-cache
> path is documented separately in `RC_SURFACE_CACHE.md` (endgame, view-independent,
> much larger build). Temporal reprojection (step A1) is needed in BOTH paths, so this
> work is not throwaway.

---

## 1. Why this fixes the blockiness

Established empirically (see the debugging session): the "pixelation" is **RC probe
undersampling + few-ray binary occlusion** → spatially banded irradiance. A wide
edge-aware blur (à-trous radius ~16) reconstructs it, but a single huge spatial kernel
is expensive and over-blurs.

The reference (`example/cube A.wgsl`) has **no spatial denoiser at all** — its only
smoothing is reading the previous frame (`iChannel3`) and accumulating. A stable cache
+ 1 ray/probe/frame, averaged over N frames, converges to clean irradiance for **free**
(amortized). We replicate that in screen space:

- Each frame still traces the same noisy few-ray RC → raw irradiance (`irA`).
- **Reproject** last frame's accumulated irradiance into this frame (by depth → world →
  previous view-projection).
- **Blend** (exponential moving average): `acc = lerp(current, history, α)`, α≈0.9.
- Feed `acc` back as next frame's history, and into the (now light) à-trous + composite.

Effective sample count ≈ `1/(1-α)` (α=0.9 → ~10 spp) at the cost of one extra full-res
pass. The à-trous can drop from 5 iterations to 1–2 (just to hide disocclusion edges).

---

## 2. What's new (data + passes)

### Textures (full G-buffer res, `rgba16float`)
- `histA`, `histB` — ping-pong **accumulated irradiance** history (read prev, write cur).
  Swapped each frame at the call site (like the engine's other ping-pongs).
- (optional, step A2) `moments` `rg16float` — luminance 1st/2nd moment for variance-driven
  à-trous + adaptive α. Skip in A1.

The existing `irA`/`irB` (gather output + à-trous ping-pong) stay. New ordering:
```
cascade loop → gather (→ irA, raw)
            → TEMPORAL accumulate (irA + reproject(histPrev) → histCur)   ← NEW
            → à-trous (histCur → … ping-pong, 1–2 iters)                  ← fewer iters
            → composite (→ out)
```
History fed back = the **temporally-accumulated, pre-spatial** irradiance (`histCur`).
(SVGF feeds back post-first-spatial-level; pre-spatial is simpler and fine for A1.)

### Uniforms
- `uReproj` block: **previous-frame** view-projection (`mat4`) + current `invViewProj`
  (already have it) + `uScreen`. Needed to map a current pixel's world point → history uv.
- `uTemporal` `vec4`: `(alpha, depthRejectRel, normalRejectDot, _)`.

### `ResizeSystem.ts` — expose previous VP
`updateViewProjection` must publish the **prior** `projectionMatrix` before overwriting:
```
export const prevViewProjection = mat4.create();
// at top of updateViewProjection, before recompute:
mat4.copy(prevViewProjection, projectionMatrix);
```
First frame: `prevViewProjection == projectionMatrix` (identity reprojection → history==current, no ghost).

---

## 3. The temporal pass (WGSL design)

Fullscreen pass, target = `histCur`, reads `irA` (current raw irradiance), `histPrev`,
`depthTex` (current), `gNormal` (current), uniforms above.

```
fn fs_main(uv):
  dC   = loadDepth(uv);  if (dC <= SKY_EPS) { return current; }   // background
  curr = sample(irA, uv).rgb;
  // 1. reconstruct THIS pixel's world pos from current depth + current invViewProj
  pW   = worldFromDepth(uv, dC);
  // 2. project with PREVIOUS frame's VP → history uv (reverse-Z aware, same as our proj)
  clip = uPrevViewProj * vec4(pW, 1);
  if (clip.w <= 0) { return current; }                            // behind prev camera
  hUV  = vec2(clip.x/clip.w*.5+.5, .5 - clip.y/clip.w*.5);
  if (any(hUV < 0 || hUV > 1)) { return current; }                // off-screen last frame
  // 3. disocclusion rejection: compare reprojected world pos depth + normal
  hist = sample(histPrev, hUV).rgb;
  // reject if the surface there is a DIFFERENT surface (moved/occluded):
  //   - world-position distance between pW and history's reconstructed world pos
  //   - normal dot below threshold
  // (cheap variant for A1: reject on normal dot + relative depth of reprojected sample)
  reject = normalMismatch(uv, hUV) || depthMismatch(...);
  a = select(uTemporal.x, 0.0, reject);                           // α=0 on reject → reset
  return mix(curr, hist, a);
```

**Reverse-Z note:** `worldFromDepth` and the projection are identical to the ones already
in `createRadianceCascades.ts` (`uv.y` flipped, reverse-Z ndc passed straight to
`invViewProj`). Reuse verbatim. The history projection uses `uPrevViewProj` (the baked
reverse-Z VP of last frame) — no remap needed, same convention.

**Dynamic-object caveat:** camera-only reprojection ghosts on moving objects (spinning
boxes, flying lights). A1 relies on **disocclusion rejection** (normal/depth mismatch
resets α→0) to break ghosts — acceptable, but moving emitters will trail slightly. A2
adds proper motion vectors (see below) to fix this cleanly.

---

## 4. Integration points

- **`createFrame.ts`**: add `histA`/`histB` to `createRCTextures` (full-res `rgba16float`,
  `RENDER_ATTACHMENT | TEXTURE_BINDING`). NB: the test scene manages cascade textures
  locally (`main.ts makeCascPair`) for live downscale — history is full-res and
  downscale-independent, so it can live in `createRCTextures` or be created by the RC
  factory in `init()` (preferred: factory owns its ping-pong, like `irA/irB`).
- **`createRadianceCascades.ts`**:
  - factory creates `histA`/`histB` in `init()` (full G-buffer res).
  - add `temporalMeta` shader (the pass above) + pipeline + 2 bind groups (read histA→write
    histB and read histB→write histA), selected by a frame-parity flag the factory tracks.
  - `run()` writes `uPrevViewProj`/`uTemporal`, inserts the temporal pass after gather,
    before à-trous; à-trous now reads the temporal output instead of `irA`.
  - keep a `frameParity` counter on the factory; swap history read/write each call.
- **`ResizeSystem.ts`**: export `prevViewProjection`, populate as in §2.
- **`main.ts`**: add GUI knobs `temporalAlpha` (0–0.98), `temporalReject` (depth/normal
  strictness), and a `temporal on/off` toggle (α=0 = off, equivalent to current behavior).
  Lower the default `denoiseIterations` to 1–2 once temporal is on.

### `RCParams` additions
```
temporalAlpha: number;       // 0 = off (no history), 0.9 ≈ 10 spp
temporalDepthReject: number; // relative depth delta that resets accumulation
temporalNormalReject: number;// min normal dot to accept history
```

---

## 5. Pitfalls / review checklist

- **Reverse-Z everywhere.** `uPrevViewProj` must be the same baked reverse-Z VP form
  (`REVERSE_Z * perspective * view`). Don't reproject with a GL-depth matrix.
- **First frame / resize:** `prevVP == curVP` → identity reproject, history == current,
  α effectively 0 visually. On canvas resize the history texture is stale-sized → recreate
  + clear (treat like a full disocclusion). The test scene's RC rebuild already reallocates;
  ensure history is reallocated too.
- **Ghosting vs lag tradeoff:** higher α = smoother but more trailing on motion. Expose it.
  Disocclusion rejection must be strict enough to avoid smearing across silhouettes.
- **Fireflies:** a bright emitter sample can persist via history. Optional neighborhood
  clamp (clamp history to the min/max of current 3×3) — standard TAA clamp; add if sparkles.
- **Don't double-count emission.** Temporal accumulates **irradiance only** (as gather
  already outputs); emission is added in composite. Keep that split.
- **HDR history:** keep `rgba16float`; accumulating in `bgra8unorm` bands badly.

---

## 6. Test plan (test scene)

1. `temporalAlpha = 0` must reproduce current output exactly (regression gate).
2. Static camera, raise α → noise/blockiness should fall frame-over-frame and converge in
   ~10–20 frames; `denoiseIterations 0` should look clean after convergence.
3. Orbit the camera → edges may shimmer briefly (disocclusion) but interiors stay clean;
   tune `temporalNormalReject`/`temporalDepthReject` so silhouettes don't smear.
4. Flying emitters → expect mild trails in A1; confirms motion-vector need for A2.
5. Perf: temporal adds one full-res pass; à-trous drops to 1–2 iters → net should be
   **cheaper** than current 5-iter à-trous AND cleaner.

---

## 7. Rollout (two sub-steps)

- **A1 — Temporal accumulation (camera-reprojection + disocclusion reject).** Everything
  above except motion vectors and moments. Fixed α, hard reject. This already removes the
  bulk of the noise for a slow/orbiting camera. à-trous → 1–2 iters.
- **A2 — Motion vectors + variance (full SVGF).** Write a velocity G-buffer channel in the
  SDF impostor pass (current-clip minus previous-clip per fragment, needs per-instance prev
  transform) so moving objects reproject correctly (no trails). Add luminance moments →
  variance estimate → drives adaptive α (more history where stable) and variance-guided
  à-trous step weights. This is the "real" SVGF; do it only if A1's motion trails matter.

> Motion vectors require per-instance **previous-frame transforms**. In the test scene
> that's trivial (we animate transforms in JS — keep last frame's matrix). In the engine
> it means double-buffering `GlobalTransform` or deriving velocity from physics; defer to A2.
