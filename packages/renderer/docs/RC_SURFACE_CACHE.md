# RC Phase 4 (Path B) — World-space Surface-Cache Radiance Cascades

> **Goal:** view-independent global illumination by caching radiance **on surfaces**
> (a probe atlas), tracing the real 3D scene, and accumulating temporally — the
> architecture of the `example/` shadertoy ("Radiance Cascades 3D"), generalized from
> its 6 hardcoded planes to our SDF-impostor instances.
>
> **Status:** design / endgame. Much larger than Path A (`RC_PHASE3_TEMPORAL.md`).
> This is essentially a mini-Lumen surface cache. Do it only if screen-space + temporal
> (Path A) proves insufficient (light leaking, view-dependence) for the game's needs.
>
> **Reference decode:** see the session analysis; the shadertoy is in `example/`
> (`common.wgsl` / `cube A.wgsl` = the cascade buffer, `image.wgsl` = primary-ray lookup).

---

## 1. Why surface-cache beats screen-space

Screen-space RC (our current Phase 2) has three structural limits that no amount of
denoising removes:
- **Light leaking at silhouettes** — single depth layer; occluders off-screen or behind
  the first surface are invisible.
- **View-dependence** — probes live on screen pixels; rotating the camera re-samples
  everything, so lighting subtly changes with view.
- **Probe density tied to pixels** — undersampling far/grazing surfaces.

Surface-cache RC fixes all three: probes live in a **surface UV atlas**, so density tracks
surface area (not pixels), the cache is **stable across views** (rotate freely, lighting is
identical), and rays are traced against the **real 3D scene** (no screen-space occlusion
artifact). The reference's smoothness comes from this stability + **temporal accumulation**
(it reads last frame's atlas as both the GI bounce source and the denoiser).

The cost: you must (a) parameterize every surface into an atlas, (b) trace the world, and
(c) manage the atlas for dynamic geometry. That's the whole build below.

---

## 2. Reference architecture (decoded, see `example/`)

The atlas (`iChannel3`) stores, per surface texel, a stack of cascades:
- Geometry chart: per surface, a `(gTan, gBit, gNor, gPos, gRes)` frame; probe at surface
  UV → `probePos = gPos + gTan·u + gBit·v`.
- `probeCascade = floor(mod(UV.y,1536)/256)` — 6 cascades stacked vertically per chart.
- `probeSize = 2^(cascade+1)` (c0=2…c5=64); `probePositions = gRes/probeSize` probes/axis;
  each probe = `probeSize²` directional bins. **Space halves, angle quadruples per cascade.**
- Directions = **hemisphere** over the normal: `θ` = ring index (Chebyshev dist in the
  probe block), `φ` = walk around the square ring, `bins_φ = 4 + 8·θ`. Built in tangent
  space, rotated by `(gTan,gBit,gNor)`.
- Ray length per cascade `tInterval = (1/64)·probeSize·2` (doubling; top = ∞ → sky).
- On hit: **radiosity bounce = read the atlas at the hit surface's c0 quad** (last frame),
  + sun shadow ray, × albedo. On miss: sky.
- Integration weight: `(cos(θ-dθ)-cos(θ+dθ))/bins_φ · cos(θ)` (solid angle × Lambert).
- **Merge = visibility-weighted bilinear:** `.w` channel stores ray hit distance; the
  interval blend `l = 1-clamp((t-minDist)/interval,0,1)` mixes own-ray vs upper cascade,
  and `WeightedSample` drops parent probes whose stored ray distance says an occluder sits
  between probes → **no interpolation light-leak**.
- Primary shading (`image.wgsl`): trace camera ray, find hit surface, read its c0 quad from
  the atlas → that's the lit color. ACES + gamma.

---

## 3. Generalizing to our SDF instances

The hard part: the reference hardcodes 6 planes. We have arbitrary, **dynamic** Box3D /
Sphere3D instances. Three subsystems to build.

### 3.1 Surface parameterization (per-instance charts)
Assign each instance a rectangular **chart** in a shared atlas, sized by surface area
(texels ∝ world area, clamped to a min/max). Mapping atlas-UV ↔ surface point + normal:
- **Box3D:** 6 faces; an octahedral or 3×2 cube-cross unwrap. Each chart texel → a face +
  in-face UV → world `pos`/`normal` via the instance transform + half-extents.
- **Sphere3D:** octahedral map (unit hemisphere/sphere → square); texel → direction →
  `pos = center + r·dir`, `normal = dir`.
- Rounded boxes: treat as box charts; the rounding is sub-texel for GI purposes.

Store per chart: instance eid, atlas rect, the inverse mapping params. This is a **chart
allocator** (the genuinely new system). For mostly-static scenery a one-time pack; for
dynamic instances a per-frame or incremental allocator (stable IDs to keep temporal history
valid — a remapped chart must invalidate its history).

### 3.2 World ray tracing
The cascade pass traces rays from `probePos` along `probeDir`. Options, cheapest→best:
- **(a) Reuse screen-space depth march** — defeats the purpose (reintroduces leaking). No.
- **(b) Analytic ray-vs-instances** — our primitives already have analytic intersections
  (`ray_box`/`ray_sphere` in `sdf.shader.ts`). Loop instances → needs a **spatial structure**
  (BVH or uniform grid in a storage buffer) to avoid O(rays·instances). Good fit for
  "many small pieces" if the grid is rebuilt cheaply each frame.
- **(c) Global SDF / voxel volume** — rasterize/splat all instances into a coarse world SDF
  (3D texture), sphere-trace it (Lumen-style). Best for many objects + soft occlusion;
  biggest build. Probably the right long-term answer for a dense tank scene.

Recommendation: start with **(b)** + a uniform grid (instances are small and numerous →
grid beats BVH to build), escalate to **(c)** if grid traversal is the bottleneck.

### 3.3 Temporal feedback
Atlas is double-buffered (`atlasPrev` / `atlasCur`). The cascade pass reads `atlasPrev`
for the radiosity bounce and the merge; writes `atlasCur`; swap each frame. This is the
denoiser **and** the multi-bounce mechanism — no spatial filter needed (the reference has
none). Moving geometry/light causes the **flickering** the reference warns about; mitigate
with per-chart history validation (invalidate on large transform delta) and/or a small
temporal clamp.

---

## 4. Pass graph (proposed)

```
0. (CPU/GPU) chart allocation + instance spatial grid build         ← new systems
1. surface main pass → G-buffer (unchanged) + per-pixel CHART ID + surface UV
2. cascade/merge atlas pass (n = 5..0):                              ← the big one
     for each atlas texel: decode (chart, cascade, probe, dir) →
       probePos/probeDir → TraceRay(grid/SDF) →
       bounce(read atlasPrev) + sun → weight → merge(atlasPrev, visibility) → atlasCur
3. (swap atlasPrev/atlasCur)
4. primary lighting pass: per screen pixel, read chartID+UV from G-buffer →
     sample atlasCur c0 quad → lit color → tonemap → present
```

Step 2 runs at atlas resolution (independent of screen res) — that's the key scaling
property (lighting cost ∝ surface area cached, not pixels).

### Data / buffers
- `atlasPrev`, `atlasCur` — `rgba16float` (rgb = radiance, a = ray hit distance).
- instance table (storage buffer): transform, half-extents/radius, albedo, emission,
  chart rect, kind.
- spatial grid (storage buffer): cell → instance id list (for ray tracing).
- chart table: eid ↔ atlas rect ↔ mapping params; + a "history valid" flag per chart.

---

## 5. ECS integration (this project's conventions)

Per CLAUDE.md (components = data, systems = behavior, query = trigger):
- **Component `SurfaceChart { atlasX, atlasY, atlasW, atlasH, historyValid }`** — present on
  instances that participate in GI. Existence-based: an instance lacking it isn't cached.
- **System `createChartAllocatorSystem`** — assigns/frees atlas rects for instances that
  gained/lost `SurfaceChart`; sets `historyValid=0` on (re)allocation.
- **System `createSurfaceCacheSystem`** — owns `atlasPrev/atlasCur`, the cascade atlas pass,
  the swap. Reads the instance + grid storage buffers.
- **System `createInstanceGridSystem`** — clears + refills the spatial grid buffer each
  frame (reused buffer, no alloc — DOD rule).
- The primary lighting pass replaces the current composite; reads `chartID`/`UV` from an
  extra G-buffer attachment written by the SDF impostor shader.

Relations are eids (chart→instance), verified before use; sentinel for "no chart". No
object refs. Keep the ray-trace loop monomorphic (branch box/sphere by a packed kind, or
two grids).

---

## 6. Risks / open questions

- **Atlas allocation for dynamic instances** is the crux. Many small moving tank parts →
  lots of charts, frequent realloc → history thrash → flicker. Needs a stable-ID allocator
  and a sensible texel budget (area-proportional, min 2×2 per chart for c0).
- **Ray-trace cost** scales with rays = atlas texels × (effectively). Atlas budget +
  cascade count bound it; the grid/SDF must be fast. Profile early.
- **Flickering** on motion (reference's known issue). Temporal validation + clamp; possibly
  reproject charts when transforms change little.
- **Memory:** two `rgba16float` atlases + instance/grid buffers. Size the atlas to a fixed
  budget (e.g. 2048² or 4096²) and pack to it.
- **Determinism / headless:** the engine runs headless for training (no GPU); GI is a
  render-only concern, so this must stay entirely in `renderer` and be skipped when
  `RenderDI` is absent. No gameplay dependence.

---

## 7. Rollout (two sub-steps)

- **B1 — Static surface cache (prove the pipeline).** Hardcode/one-time chart packing for
  the test scene's static instances (ground, pillars, spheres). Analytic ray-vs-instances
  with a uniform grid (option b). Temporal feedback + atlas. Goal: view-independent GI on a
  static scene, matching the reference's quality, no leaking. This validates the atlas
  layout, hemisphere encoding, visibility-weighted merge, and temporal feedback end-to-end.
- **B2 — Dynamic charts + scale.** The chart allocator for spawning/moving/dying instances
  with history invalidation; escalate ray tracing to a global SDF (option c) if the grid is
  the bottleneck; flicker mitigation. This is what makes it usable in the actual game.

> Before starting B at all: confirm Path A (temporal screen-space) is genuinely
> insufficient. B is a large, ongoing subsystem; A is a few passes. Borrow A's temporal
> reprojection learnings — the accumulation math is the same idea, just in a different domain.
