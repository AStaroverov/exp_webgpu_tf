# Radiance Cascades 2D Lighting — Integration Plan

## Overview & goal

Add a 2D global-illumination (GI) lighting pass to the WebGPU renderer using the
**Radiance Cascades** (RC) technique. RC computes soft, colored, omnidirectional
light propagation with penumbrae and bounce-like fill, driven by an _emission map_
(colored light sources) and an _occluder coverage map_ (everything solid). This
replaces — or, during bring-up, coexists with — the current single fixed-direction
height-shadow system baked into `sdf.shader.ts`.

The lighting pass is **visualization-only**. Headless RL rollouts never create a
render target (`RenderDI` is absent), so RC has zero cost on training throughput; it
only competes with the live debug-tab view (one episode at a time, sharing the GPU
with WebGPU-TF learners). Budget is generous — correctness and look matter more than
per-frame cost.

**Reference implementation:** radiance-cascades.com single-file WebGL2/GLSL demo.

- Online: https://github.com/radiance-cascades/radiance-cascades.com/blob/main/public/js/rc.js
- Local copy: `/tmp/rc.js`

The reader is expected to read `rc.js` for shader detail; this document only
summarizes the algorithm and focuses on the **engine integration**.

---

## Algorithm summary

The reference pipeline (class chain `BaseSurface → Drawing → JFA → DistanceField → RC`)
runs, every frame, six stages:

1. **Scene / emission** — produce a color texture: `RGB` = emitted/material color
   (premultiplied), `A` = coverage (`> 0` means a surface is present).
2. **Seed** — for each texel with `alpha > 0`, store its own UV as a seed; else `(0,0)`.
3. **JFA (Jump Flood)** — `ceil(log2(maxDim)) + 1` ping-pong passes propagate the
   nearest-seed UV outward (offset halves each pass: `2^(passes-1) … 1`), producing a
   nearest-seed Voronoi map.
4. **Distance field (DF)** — `clamp(distance(vUv, nearestSeed), 0, 1)` → scalar field
   in normalized UV units.
5. **Radiance cascades** — for `cascadeIndex` from `cascadeCount-1` down to `0`:
   each probe casts `baseRayCount` rays over a radial interval, sphere-traces the DF,
   samples scene color on hit (sRGB-decoded), and **merges** missed rays with the
   matching direction from the cascade above (bilinear interpolation across the upper
   probe grid). Higher cascades = fewer probes (coarser space), more rays (finer
   angle), longer intervals. Cascade 0 is the visible result.
6. **Overlay / composite** — combine cascade-0 radiance with the scene to screen.

Key formulas to preserve exactly (see `/tmp/rc.js` and §"GLSL→WGSL porting notes"):

- `cascadeCount = ceil(log(diag) / log(baseRayCount)) + 1`, `diag = sqrt(w² + h²)`.
- `spacingBase = sqrt(baseRayCount)`, `spacing = pow(spacingBase, cascadeIndex)`.
- Interval: `start = (idx==0 ? 0 : base^(idx-1)) * modInterval`,
  `end = ((1 + 3*intervalOverlap) * base^idx - idx²) * modInterval`.
- `angle = (index + 0.5 + noise) * angleStep`, `rayDir = vec2(cos, -sin)` (**y-flip**).
- Merge early-out: `currentRadiance.a > 0 || cascadeIndex >= max(1, cascadeCount-1)`.

We **do not** port the "reduce-demand" two-frame split (`forceFullPass = true`
always), the brush/UI/SDF authoring, the GPU timer, or the WebGL micro-layer.

---

## Final architecture

### Decision: fragment fullscreen passes, not compute

The engine is entirely fullscreen-fragment + instanced-draw. `GPUShader` exposes only
`getRenderPipeline` — there is **no** `getComputePipeline`, no storage-texture write
path, and no mip-generation helper. RC is implemented as a sequence of fullscreen
fragment passes, each shaped like the existing `createPostEffect` (a `GPUShader`, a
sampler, `draw(6, 1, 0, 0)` over the hardcoded 6-vertex quad). The JFA loop is the
natural future compute candidate — leave a one-line note, do not build it now.

`basePixelsBetweenProbes` is **fixed at 1**, which forces the merge LOD term to `0.0`
and lets us **drop mip generation entirely** — the single most important simplifying
decision, since the engine has no mip-gen path.

### Pass diagram

```
                        per-frame, on the single commandEncoder from createGame.renderFrame
                        ─────────────────────────────────────────────────────────────────
  [createFrameTick]
   shadow-map pass ── own encoder + submit (existing hack — DO NOT copy)
   main pass ───────► renderTexture (bgra8unorm, scene color, already shadowed/lit-base)

  [createRadianceCascadesSystem.run(encoder)]   ← NEW, between frameTick and postEffect
   emit pass     drawEmitters (SDF vs_emit/fs_emit, additive) ─► emissionTexture (rgba16f)
   seed pass     emissionTexture.a                            ─► seedA (rg16f)
   jfa pass ×N   ping-pong seedA⇄seedB, offset 2^(p-1)…1      ─► seedA/seedB (rg16f)
   df pass       nearest-seed map                             ─► dfTexture (r16f)
   rc pass ×C    cascadeIndex C-1…0, ping-pong cascA⇄cascB,   ─► cascA/cascB (rgba16f)
                 each reads dfTexture + emissionTexture + prev cascade
   composite     scene * (ambient + cascade0)                 ─► litTexture (bgra8unorm)

  [createPostEffect(litTexture)]                ← repointed from renderTexture (1-line)
   pixelate pass litTexture                                   ─► swapchain
```

All RC passes record into the **single** `commandEncoder` created in
`createGame.renderFrame`, with **no per-pass `submit`**. WebGPU orders passes on one
encoder correctly via the implicit texture-state barriers; the shadow-map pass's
separate-encoder + immediate-submit is a legacy hack we explicitly do not replicate.

### Textures

Added to `createFrameTextures` in `packages/renderer/src/WGSL/createFrame.ts`, all
sized `rcW × rcH` where `rcW = floor(canvas.width * rcDownscale)` (default `1.0`).

| Name              | Format        | Size      | Filter  | Sampled via          | Usage                                                    |
| ----------------- | ------------- | --------- | ------- | -------------------- | -------------------------------------------------------- |
| `emissionTexture` | `rgba16float` | rcW × rcH | linear  | `textureSampleLevel` | RA \| TB — RGB = premult HDR emitter color, A = coverage |
| `seedA`           | `rg16float`   | rcW × rcH | nearest | `textureLoad`        | RA \| TB — JFA ping (seed UV)                            |
| `seedB`           | `rg16float`   | rcW × rcH | nearest | `textureLoad`        | RA \| TB — JFA pong                                      |
| `dfTexture`       | `r16float`    | rcW × rcH | nearest | `textureLoad`        | RA \| TB — distance field (UV units)                     |
| `cascA`           | `rgba16float` | rcW × rcH | linear  | `textureSampleLevel` | RA \| TB — cascade ping (radiance)                       |
| `cascB`           | `rgba16float` | rcW × rcH | linear  | `textureSampleLevel` | RA \| TB — cascade pong                                  |
| `litTexture`      | `bgra8unorm`  | canvas    | nearest | `textureSample`      | RA \| TB — composite output, fed to Pixelate             |

RA = `RENDER_ATTACHMENT`, TB = `TEXTURE_BINDING`. **Single mip on all** (no
mip-gen, because `basePixelsBetweenProbes = 1`). We use `rgba16float` rather than the
reference's `r11f_g11f_b10f` — the latter is not guaranteed renderable + filterable +
blendable in WebGPU. `rg16float` / `r16float` / `rgba16float` are all filterable, so
`textureSample*` works; the seed/JFA/DF passes still use `textureLoad` (no filtering
desired there). Avoid 32-bit float formats — those are unfilterable in this engine
(the existing shadow map at `r32float` is read via `textureLoad` for exactly this
reason).

### Data flow

```
SDF instances (Color, GlobalTransform, Shape, LightEmitter)
      │  drawEmitters (reuses SDF instance pipeline, additive blend)
      ▼
emissionTexture ──.a──► seed ──► JFA×N ──► dfTexture
      │                                        │
      └────────────────── rc cascade loop ◄────┘   (reads emission + DF + prev cascade)
                                  │
                            cascade0 radiance
                                  │
   renderTexture (scene) ── composite: scene*(ambient+radiance) ──► litTexture ──► Pixelate ──► swapchain
```

`renderTexture` remains the full scene color (drawn by the existing
`fauna → grid → shapes → vfx → sandstorm` order). The composite pass reads both
`renderTexture` and `cascA`/`cascB` cascade-0 output and multiplies the scene by
`(ambient + radiance)`.

---

## ECS integration

### New component: `LightEmitter`

File: `packages/renderer/src/ECS/Components/Common.ts` (follows the existing
`createColorComponent` / `createRoundnessComponent` `defineComponent`/`obs` pattern).

```ts
export const createLightEmitterComponent = defineComponent((LightEmitter, obs) => {
  const intensity = TypedArray.f64(delegate.defaultSize); // emission multiplier; 0 = pure occluder
  const radius = TypedArray.f64(delegate.defaultSize); // optional falloff hint
  return {
    intensity,
    radius,
    addComponent(world, eid, i = 1, r = 0) {
      addComponent(world, eid, LightEmitter);
      intensity[eid] = i;
      radius[eid] = r;
    },
    set$: obs((eid, i, r) => {
      intensity[eid] = i;
      radius[eid] = r;
    }),
  };
});
```

Register it in `createRenderComponents` in
`packages/renderer/src/ECS/world.ts` (next to `Color`, `Roundness`, `Shape`):

```ts
LightEmitter: createLightEmitterComponent(world),
```

**No `Occluder` component.** Coverage = any drawn SDF alpha. _Every_ SDF shape
occludes; shapes tagged `LightEmitter` additionally emit. Emission encoding reuses the
existing `Color` component: emitted color = `Color.rgb * intensity`, `A = coverage`.
Occluders (no emitter, or `intensity = 0`) write `RGB = 0, A = 1`. This avoids adding
a new SoA channel to the change-detected upload path in `createDrawShapeSystem`.

### Emission draw pass: reuse the SDF instance pipeline

Add `vs_emit` / `fs_emit` entry points to
`packages/renderer/src/ECS/Systems/SDFSystem/sdf.shader.ts` and a `drawEmitters(pass)`
callback + `pipelineEmit` in
`packages/renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts`, mirroring the
existing `vs_shadow_map` / `fs_shadow_map` subpass exactly:

- `vs_emit` = `vs_main` (same instanced quad, same `to_final_position`).
- `fs_emit` evaluates `sd_shape`, `discard`s outside the shape, and outputs
  `vec4(Color.rgb * intensity, 1.0)` inside (intensity supplied via the `LightEmitter`
  uniform/storage already bound, or `1.0` if absent — occluders pass `intensity = 0`).
- `pipelineEmit`: `getRenderPipeline(device, 'vs_emit', 'fs_emit', { autoLayout: true,
bindGroups: {...}, targetFormat: 'rgba16float', withDepth: false, blend: 'additive' })`.

This gives perfect spatial registration with the scene (same transforms, same SDF
evaluation) and reuses the SoA upload that already runs for `drawShapes`.

### New lighting system (game-side)

New directory:
`packages/unknown/src/Game/ECS/Systems/Render/Lighting/`

```
createRadianceCascadesSystem.ts   orchestrator
seed.shader.ts                    ShaderMeta + WGSL (seed)
jfa.shader.ts                     ShaderMeta + WGSL (jump flood)
df.shader.ts                      ShaderMeta + WGSL (distance field)
radianceCascades.shader.ts        ShaderMeta + WGSL (cascade pass)
overlay.shader.ts                 ShaderMeta + WGSL (composite)
```

`createRadianceCascadesSystem({ world, device, textures, drawEmitters, params? })`
returns `{ run(encoder, delta), outputTexture: litTexture, recreate(canvas), destroy }`,
modeled on `createPostEffect` / `createDrawShapeSystem`. `createFrameTick` stays
**untouched** — RC is a standalone orchestrator invoked from `createGame.renderFrame`.

### Wiring (`packages/unknown/src/Game/createGame.ts`, ~278–317)

The single wiring point. Inside `setRenderTarget`:

```ts
const textures = createFrameTextures(device, canvas); // now also has RC textures
const shapeSystem = createDrawShapeSystem({
  world,
  device,
  shadowMapTexture: textures.shadowMapTexture,
});
const lighting = createRadianceCascadesSystem({
  world,
  device,
  textures,
  drawEmitters: shapeSystem.drawEmitters,
});
const frameTick = createFrameTick(
  { ...textures, canvas, device, background, getPixelRatio },
  mainCb,
  shadowMapCb,
);
const postEffectFrame = createPostEffect(device, context, lighting.outputTexture); // was textures.renderTexture

RenderDI.renderFrame = (delta) => {
  const commandEncoder = device.createCommandEncoder();
  frameTick(commandEncoder, delta); // scene → renderTexture (+ shadow pass, own submit)
  lighting.run(commandEncoder, delta); // emit/seed/jfa/df/cascades/composite → litTexture
  postEffectFrame(commandEncoder); // litTexture → swapchain
  device.queue.submit([commandEncoder.finish()]);
};
```

The composite stays fully self-contained in `/Lighting/`. Reverting RC is one line
(repoint `createPostEffect` back at `textures.renderTexture`); the Pixelate shader is
**not** modified.

### Emitter tagging (game content)

Tag entities with `LightEmitter` where light should radiate:

- `packages/unknown/src/Game/Config/spice.ts` — spice (warm melange glow).
- VFX entities — `Explosion`, `MuzzleFlash`, `HitFlash` (strong transient emitters).
- Bullets / tracers — moving point lights.
- Occluders (`Config/obstacles.ts` rocks/walls, tank bodies) get `intensity = 0` or
  simply no `LightEmitter` — they already occlude via their SDF alpha.

---

## Renderer-side change: additive blend + cache-key fix

The emission pass needs **additive** blending (overlapping emitters accumulate).
`getRenderPipeline` currently hardcodes only standard alpha blend
(`src-alpha / one-minus-src-alpha`, lines 69–80) and there is no additive option.

Extend `getRenderPipeline` (`packages/renderer/src/WGSL/GPUShader.ts`) with a blend
selector, e.g. `blend?: 'alpha' | 'additive'` mapping additive to
`src = one, dst = one, op = add`.

**Critical:** the pipeline cache key (line 50) is currently
`` `${vertexName}-${fragmentName}-${withDepth}-${targetFormat}-${withBlending}-${autoLayout}` ``
and does **not** include the blend variant (nor `bindGroups`). The new blend mode
**must be added to the key**, otherwise an additive pipeline and an alpha pipeline
with the same `vs/fs/format/withDepth/autoLayout` tuple collide in the cache (first
call wins → silent wrong blend). The emission pass uses distinct entry points
(`vs_emit/fs_emit`), so the `bindGroups` omission is not hit here, but the blend
dimension is — extend the key.

---

## GLSL → WGSL porting notes

Port these from `/tmp/rc.js` **verbatim in logic** (only the language changes):
`raymarch`, `getUpperCascadeTextureUv` + `merge`, the rc `main()` probe/ray loop, the
JFA `classic()` kernel, the seed and DF shaders, and optionally `sunAndSky` /
`oldSunAndSky`.

Mechanical translation rules:

| GLSL                                            | WGSL                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| `texelFetch(tex, ivec2, 0)`                     | `textureLoad(tex, vec2<i32>(...), 0)` — seed/JFA/DF (no sampler)   |
| `texture(tex, uv)` / `textureLod(tex, uv, 0.0)` | `textureSampleLevel(tex, samp, uv, 0.0)` — raymarch + merge        |
| `mod(a, b)`                                     | `a - b * floor(a / b)` (helper `fn fmod`)                          |
| `cond ? a : b`                                  | `select(b, a, cond)`                                               |
| `for (float dist=0.; dist<len;)`                | bounded loop with `MAX_STEPS` cap (no infinite march)              |
| `vec4 FragColor`                                | `@location(0) vec4f`; extra channels dropped by `rg16/r16` targets |

**`textureSampleLevel` is mandatory inside the raymarch loop.** `textureSample` is
illegal in WebGPU under non-uniform control flow (the conditional, data-dependent
march), so every in-loop fetch must use the explicit-LOD variant. This is the single
biggest mechanical gotcha.

### Key snippets to port

Fullscreen quad: copy the `POSITION` / `TEX_COORDS` arrays and `vs_main` from
`packages/unknown/src/Game/ECS/Systems/Render/PostEffect/Pixelate/shader.ts` for every
RC pass. **Pin the orientation at milestone M1** — the reference's
`rayDir = vec2(cos, -sin)` (y-flip) must be reconciled against the engine quad's
flipped `TEX_COORDS` V before any JFA/raymarch work, or every downstream pass is
upside-down.

Seed:

```glsl
float alpha = texelFetch(surfaceTexture, ivec2(gl_FragCoord.xy), 0).a;
FragColor = vUv * ceil(alpha);
```

Distance field:

```glsl
vec2 nearestSeed = texelFetch(jfaTexture, texel, 0).xy;
FragColor = clamp(distance(vUv, nearestSeed), 0.0, 1.0);
```

Raymarch (sphere-trace the DF; sRGB-decode scene on hit):

```glsl
vec2 rayDir = normalize(rayEnd - rayStart);
float rayLength = length(rayEnd - rayStart);
vec2 rayUv = rayStart * oneOverSize;
for (float dist = 0.0; dist < rayLength;) {
  if (any(lessThan(rayUv, vec2(0.0))) || any(greaterThan(rayUv, vec2(1.0)))) break;
  float df = textureLod(distanceTexture, rayUv, 0.0).r;
  if (df <= minStepSize) {
    vec4 s = textureLod(sceneTexture, rayUv, 0.0);
    s.rgb = pow(s.rgb, vec3(srgb));
    return s;
  }
  dist  += df * scale;
  rayUv += rayDir * (df * scale * oneOverSize);
}
return vec4(0.0);
```

Merge (the bilinear-fix upper fetch):

```glsl
if (currentRadiance.a > 0.0 || cascadeIndex >= max(1.0, cascadeCount - 1.0))
  return currentRadiance; // hit, or top cascade: nothing to merge
vec2 offset = (position + localOffset) / spacingBase;
vec2 upperUv = getUpperCascadeTextureUv(index, offset, spacingBase);
vec3 upper = textureLod(lastTexture, upperUv, 0.0).rgb; // LOD 0 because basePixelsBetweenProbes==1
return currentRadiance + vec4(upper, 1.0);
```

Cascade `main()` interval + ray loop (preserve every magic number):

```glsl
float rayCount = pow(base, cascadeIndex + 1.0);
float spacing  = pow(sqrt(base), cascadeIndex);
float start = (cascadeIndex == 0.0 ? 0.0 : pow(base, cascadeIndex - 1.0)) * modInterval;
float end   = ((1.0 + 3.0*intervalOverlap) * pow(base, cascadeIndex) - pow(cascadeIndex,2.0)) * modInterval;
// ... per ray:
float angle = (index + 0.5 + noise) * angleStep;
vec2 rayDir = vec2(cos(angle), -sin(angle));   // y-flip
vec2 rayStart = probeCenter + rayDir * start;
vec2 rayEnd   = rayStart   + rayDir * end;      // NOTE: end added AFTER start offset
```

JFA `classic()`: sample center + 8 neighbors at `±uOffset * oneOverSize`, keep the
non-zero seed with minimum squared distance to `vUv`; write the seed UV. Translate the
nested `for (x,y)` loop directly.

**sRGB:** default `srgb = 2.2`; decode scene on raymarch hit (`pow(rgb, srgb)`); encode
only on the final visible cascade (`cascadeIndex <= firstCascadeIndex`,
`pow(rgb, 1/srgb)`). `bgra8unorm` is **not** a hardware `*_srgb` format, so in-shader
gamma is correct here. Keep `srgb` a param so `srgb = 1.0` is an escape hatch if the
composite (multiplying radiance into LDR display-space color) double-darkens.

---

## Parameters & defaults

`RCParams`, all radial/probe quantities derived from `rcW, rcH` and recomputed in
`recreate()`:

| Param                     | Default         | Notes                                                    |
| ------------------------- | --------------- | -------------------------------------------------------- |
| `baseRayCount`            | `4`             | angular rays per probe                                   |
| `basePixelsBetweenProbes` | `1` (**fixed**) | forces merge LOD 0 → no mips                             |
| `rayInterval`             | `1.0`           | radial interval scale                                    |
| `intervalOverlap`         | `0.1`           | overlap term in `end`                                    |
| `cascadeInterval`         | `1.0`           | hardcoded in reference                                   |
| `srgb`                    | `2.2`           | `1.0` = escape hatch                                     |
| `enableSun`               | `false`         | sun/sky injection at top cascade                         |
| `firstCascadeIndex`       | `0`             | visible cascade / sRGB-encode branch                     |
| `rcDownscale`             | `1.0`           | run RC at e.g. `0.5×` on high-DPR, upsample in composite |
| `ambient`                 | `~0.15`         | composite floor so unlit areas aren't pure black         |

Derived: `cascadeCount = ceil(log(sqrt(rcW²+rcH²)) / log(baseRayCount)) + 1`;
JFA `passes = ceil(log2(max(rcW, rcH))) + 1`.

Composite math (in `overlay.shader.ts`): `out = sceneColor * (ambient + radiance)` —
a game-appropriate replacement for the reference's hard "show surface else radiance"
overlay, keeping the scene visible in shadow rather than blacking it out.

---

## Incremental milestone plan

Each milestone is independently verifiable on screen in the debug tab.

- **M0 — Plumbing + resize + emission.** Add RC textures to `createFrameTextures` and a
  `recreate(canvas)` on the lighting system wired to the same size-change signal
  `ResizeSystem` detects (this also fixes the pre-existing resize bug for the new
  textures). Add `vs_emit/fs_emit` + `drawEmitters` + additive blend (+ cache-key fix).
  **Verify:** blit `emissionTexture` to screen — emitters glow as colored blobs at the
  right positions; occluders are dark.
- **M1 — Seed (PIN ORIENTATION).** Seed pass → `seedA`. **Verify:** visualize seed UV
  as RG — surfaces show their own UV, empty space is black, and the image is _right way
  up_. Reconcile the `(cos, -sin)` y-flip vs quad V here before proceeding.
- **M2 — JFA + DF.** JFA loop + DF pass. Highest-bug-risk stage. **Verify:** the DF
  grayscale ramps smoothly from 0 at surfaces to 1 far away — this is the key
  correctness signal for the whole pipeline.
- **M3 — Single top cascade.** Run only `cascadeIndex = cascadeCount-1`, output its
  radiance directly. **Verify:** raymarching + occlusion work — light stops at
  occluders, correct direction, no y-flip.
- **M4 — Full merge.** Full cascade loop high→low with `merge`. **Verify:** smooth
  penumbrae and bounce fill; check for seams between cascade tiles.
- **M5 — Composite + wire + gamma check.** Composite `scene * (ambient + radiance)` →
  `litTexture`, repoint `createPostEffect`. **Verify:** lit scene on screen; confirm
  colors aren't muddy/double-darkened (flip `srgb = 1.0` if so).
- **M6 — (optional) Retire directional shadow.** In a **separate** change, remove
  `vs/fs_shadow`, `vs/fs_shadow_map`, `shadowMapTexture`, and the shadow pass's
  separate-encoder submit. Never couple this with the RC-add change.

---

## Risks & performance notes

- **Resize (top risk).** `createFrameTextures` creates textures **once** and never
  recreates them on resize, while the canvas _is_ resized — a latent bug today. The new
  RC textures inherit it. Handle in M0 via `recreate(canvas)`; recompute
  `cascadeCount`/resolution and rebuild dependent bind groups + textures on the
  `ResizeSystem` size-change signal.
- **Pipeline cache-key collision.** Adding additive blend without extending the cache
  key (`GPUShader.ts:50`) silently reuses the wrong pipeline. Extend the key.
- **`textureSample` in non-uniform control flow** is invalid — use
  `textureSampleLevel(..., 0.0)` in the raymarch and merge loops.
- **Orientation.** The `(cos, -sin)` y-flip vs the engine quad's flipped `TEX_COORDS`
  V; pin at M1 or everything is upside-down.
- **Gamma.** Compositing radiance into the LDR `renderTexture`/`litTexture` can
  double-darken; keep `srgb` a param.
- **Format portability.** Use `rgba16float` (renderable + filterable + blendable), not
  `r11f_g11f_b10f`; avoid 32-bit floats (unfilterable in this engine).
- **Perf.** RC is visualization-only (no headless cost) but shares the GPU with
  WebGPU-TF learners in the debug tab. `cascadeCount` grows with the diagonal (~6 at
  512²) and each cascade is a full-res fragment pass; `rcDownscale = 0.5` halves RC
  cost and upsamples cleanly via the composite's linear sampler — preferred over
  `basePixelsBetweenProbes = 2` (which would reintroduce mips).

### Deferred / dropped (anti-speculation)

- **Do not** extract a generic `runFullscreenPass` helper up front — each pass is a
  `createPostEffect`-shaped closure; extract only if duplication actually hurts after M4.
- **Do not** add mip generation, `getComputePipeline`, storage-texture support, or
  `rg32float` seed up front. `rg32float` seed is the isolated escape hatch _only_ if
  JFA artifacts appear at high DPR (seed/JFA already use `textureLoad`, so no filtering
  is lost by switching).
- **Do not** add a separate `Occluder` component or a new `emitIntensity` SoA channel —
  coverage = SDF alpha, emission color = `Color.rgb * intensity`.

### Key files to touch

- `packages/renderer/src/WGSL/createFrame.ts` — RC textures + resize recreate.
- `packages/renderer/src/WGSL/GPUShader.ts` — additive blend + cache-key extension.
- `packages/renderer/src/ECS/Components/Common.ts` + `packages/renderer/src/ECS/world.ts` — `LightEmitter`.
- `packages/renderer/src/ECS/Systems/SDFSystem/sdf.shader.ts` + `createDrawShapeSystem.ts` — `vs_emit/fs_emit`, `drawEmitters`, `pipelineEmit`.
- **NEW** `packages/unknown/src/Game/ECS/Systems/Render/Lighting/` — `createRadianceCascadesSystem.ts` + `seed/jfa/df/radianceCascades/overlay` `.shader.ts`.
- `packages/unknown/src/Game/createGame.ts` (~278–317) — create lighting system, run on the shared encoder, feed `litTexture` to `createPostEffect`.
- `packages/unknown/src/Game/Config/spice.ts` + VFX/bullet entities — emitter tagging.
