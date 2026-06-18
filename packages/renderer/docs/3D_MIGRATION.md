# 3D migration — renderer (Phases 0–1)

> Goal: turn the 2D instanced-SDF renderer into a 3D one that draws **many small
> rigid pieces** (tank parts, projectiles, particles) as **per-instance SDF
> impostors** over a small primitive set (**box, sphere** — analytic ray intersection),
> lit **entirely by Radiance Cascades** (no shadow-map). This doc covers the two
> phases that are firmly decided and implementable now. RC itself (Phase 2) is a
> separate design; here we only fix the **contract** Phase 1 must satisfy for it.

## Decisions (locked)

- **Render path:** per-instance impostors. Each instance draws its **bounding box**
  (a cube); the fragment shader finds that instance's surface in **local space**
  (analytic for sphere + sharp box, sphere-trace for rounded box), writes a
  reconstructed depth + a G-buffer. O(1) per fragment per instance — same scaling
  property the 2D quad+SDF has today.
- **Primitives:** `box` and `sphere` only (https://iquilezles.org/articles/distfunctions/).
  A flat tile / "plane" is just a thin `box` (one near-zero half-extent) — there is no
  separate plane primitive (an infinite plane impostor needs a huge bounding cube and
  surprises as an everything-covering floor; dropped).
- **Intersection is analytic, not sphere-traced** (decided after sphere tracing starved
  on grazing rays across large flat surfaces — the ground rendered as a wedge). `box`
  (sharp), `sphere` are intersected in closed form (1 step, exact normal, no grazing
  failure). Sphere tracing survives **only** as the fallback for **rounded** boxes
  (`roundness > 0`), which have no cheap closed form and are small enough to hit head-on.
- **Lighting:** screen-space Radiance Cascades over the G-buffer (Phase 2). The
  existing `vs_shadow_map`/`fs_shadow_map` and `vs_shadow`/`fs_shadow` passes are
  **deleted**, not ported.
- **Phase 1 must output a G-buffer** (depth + world-normal + albedo + emission), not a
  final-color image. That is the contract Phase 2 consumes.

## What carries over unchanged

These are already 3D-ready and must NOT be rewritten:

- **`LocalTransform`/`GlobalTransform` are `mat4`** (`Transform.ts`) and
  `TransformSystem` already does `mat4.multiply` hierarchy → 3D parenting works as-is.
- **Instancing architecture**: storage-buffer-per-attribute + `ChangedDetector` +
  `GPUShader` pipeline/bind-group factory (`createDrawShapeSystem.ts`,
  `GPUShader.ts`). The migration swaps the *quad+2D-SDF* kernel for a *cube+3D-SDF*
  kernel; the upload/batching machinery stays.
- **Depth buffer**: `depth32float`, `depthClearValue: 0`, `compare: greater-equal`
  (`createFrame.ts`) — this is **reverse-Z** and is exactly what we want for
  perspective precision. Keep it.

---

## Phase 0 — make the engine actually 3D

Renderer-only; no gameplay changes. Goal: a perspective camera and a valid 3D
transform/depth path, verified with the *existing* 2D shapes still drawing (treated as
flat quads at z=0) before Phase 1 swaps the kernel.

### 0.1 Camera → view-projection (`ResizeSystem.ts`)

Today `projectionMatrix` is `mat4.ortho`. Replace with a combined **view-projection**:

- `projection = mat4.perspective(fovy, aspect, near, far)` — **reverse-Z**: swap
  near/far (or post-multiply a z-flip) so `near → 1`, `far → 0`, matching the existing
  `clear 0 + greater-equal`.
- `view = mat4.lookAt(eye, target, up)`.
- Export the product as `viewProjection` (keep the name `projectionMatrix` if you want
  zero churn in consumers — the shader does `uProjection * model`, which becomes
  `VP * model`, correct).
- **Remove the ad-hoc Y-flip** `return vec2(res.x, -res.y)` in the shader
  (`to_final_position`, `project_to_clip`). Bake handedness into `lookAt`'s `up` and the
  projection; do not flip in two places.

New camera API (replaces pan/zoom): `setCameraEye/Target/Up`, or an orbital/follow
controller (`yaw, pitch, distance, target`). Recommended: orbit + follow-target, since
gameplay is vehicle-centric.

Also export, for Phase 1's ray reconstruction:
- `uCameraPos: vec3<f32>` (eye in world space),
- the VP matrix (already a uniform), and optionally `uInvViewProj` if reconstructing
  rays from clip instead of from interpolated world position.

### 0.2 Transform helpers (`Transform.ts`)

- Keep `mat4`. Gameplay can keep yaw-only (`setMatrixRotateZ`) initially — tank yaw is
  Z rotation and stays valid in 3D.
- Add full 3D compose helpers when needed (quaternion or `rotateX/Y/Z`). Not required
  for Phase 0 to run.
- `m[14]` (z) is now a real world axis, no longer a shadow hack.

### 0.3 Frame loop (`createFrame.ts`)

- **Delete the shadow-map pass** (the whole first `device.queue.submit` block) and the
  `shadowMapTexture` + `shadowMapCallback` plumbing.
- Main pass keeps depth. Phase 0 still renders the old 2D shapes (flat) just to confirm
  perspective + depth + camera are correct. Phase 1 changes what the pass draws.

### Phase 0 done = perspective camera orbits a scene of (still-2D) shapes, depth sorts
correctly, no shadow-map code remains.

---

## Phase 1 — SDF impostor pipeline (the new kernel)

Replace the quad+2D-SDF draw with cube+3D-SDF impostors that write a G-buffer.

### 1.1 Geometry: a cube per instance, not a quad

- Vertex shader generates a **unit cube** (36 verts / 12 tris) from `vertex_index`
  procedurally — same "no vertex buffer" style as today's `compute_rect_vertex`.
  `renderPass.draw(36, preparedEntityCount)`.
- Cube is scaled to the instance's **bounding box** (from `Shape.values`: box
  half-extents, sphere radius, plane extents) then transformed by `model` and `VP`.
- **Cull front faces** (`cullMode: 'front'`): rasterize the *far* faces so fragments
  still exist when the camera is near/inside the box; march the ray from cube entry.

### 1.2 Fragment: analytic ray-primitive intersection

Per fragment:
1. Reconstruct the **world-space ray**: `origin = uCameraPos`,
   `dir = normalize(worldPos - uCameraPos)` (worldPos = interpolated cube-surface
   position from the vertex stage).
2. Transform ray into **instance-local space** via `uInvTransform[i]` (origin and
   dir), so the primitive stays canonical (centered, axis-aligned).
3. **Find the surface per kind:** sphere + **sharp** box → analytic closed form
   (`ray_sphere` / `ray_box`): 1 step, exact normal, no grazing failure. **Rounded**
   box → IQ-style sphere-trace (analytic AABB entry + relative epsilon + ~96 steps),
   safe because rounded boxes are small/compact. `discard` on miss.
   (A naïve sphere-trace was tried first and failed: marching a *huge flat ground box*
   from the camera with a fixed epsilon starved on grazing rays — the far ground dropped
   out. The fix was twofold: big flat surfaces are sharp boxes → analytic; the rounded
   march starts at the AABB entry and uses a relative epsilon, like IQ's gallery.)
4. Transform the local normal to world with the **normal matrix** (inverse-transpose of
   `model` upper-3×3 == transpose of `uInvTransform` upper-3×3).
5. Compute **world hit position** → `clip = VP * vec4(worldHit,1)` →
   `frag_depth = clip.z / clip.w` (reverse-Z compatible).
6. Write the **G-buffer** (1.4).

### 1.3 Primitives — `ray_box` / `ray_sphere`

`ShapeKind` is hard-cut to 3D kinds: `Box3D`, `Sphere3D` (the 2D kinds are gone). A flat
tile / "plane" is a **thin Box3D** (one near-zero half-extent) — no separate plane
primitive (an infinite-plane impostor needs a huge bounding cube and reads as an
everything-covering floor).

- **box (sharp, `roundness == 0`)**: analytic ray-AABB (slab test) → entry `t` +
  axis-aligned face normal.
- **box (rounded, `roundness > 0`)**: sphere-trace `sd_rbox = sd_box(core) - r`
  (`core = he - r`) between the analytic AABB enter/exit `t`, relative epsilon
  `d < ROUND_REL_EPS * t`, normal via tetrahedron gradient. The impostor cube extent is
  still `he` (rounded faces reach `he`; only corners round in).
- **sphere**: ray-sphere quadratic → nearest positive root + `normalize(hit)` normal.

`Shape.values` layout (6-float row):
- box: `(hx, hy, hz)` half-extents
- sphere: `(r)` radius

`Roundness` (the existing component) drives the rounded-box corner radius `r`.

### 1.4 G-buffer (the Phase 2 contract)

The main pass becomes **MRT** (the `GPUShader` factory already supports `targets[]` —
see the current emit pipeline). Attachments:

| # | name      | format      | contents                                  |
|---|-----------|-------------|-------------------------------------------|
| 0 | albedo    | `rgba8unorm`| base color (`uColor`)                     |
| 1 | normal    | `rgba16float`| world-space normal (or octahedral in `rg`)|
| 2 | emission  | `rgba16float`| HDR emitter color (`uColor*abs(intensity)`)|
| – | depth     | `depth32float`| reconstructed ray-hit depth (reverse-Z)  |

World **position** is reconstructed in the RC pass from depth + `uInvViewProj` — no
need to store it. This subsumes today's separate emission pass: emission is just G-buffer
attachment 2 written in the same draw.

### 1.5 CPU prepare changes (`createDrawShapeSystem.ts`)

- Keep `transformCollect` (model matrices).
- **Add `invTransformCollect`** — `mat4.invert(model)` per instance, uploaded as a new
  `uInvTransform` storage array. (Normal matrix can be derived in-shader from it, or
  uploaded separately.)
- `kind`/`values` updated for 3D primitives.
- Upload `uCameraPos` + VP (+ `uInvViewProj` if used).
- Drop all shadow-map-specific collects/uploads.

### 1.6 Stopgap lighting (so Phase 1 is verifiable before RC)

Add a trivial composite pass: `albedo * (ambient + saturate(dot(N, sunDir)))` reading
the G-buffer → present. Throwaway, replaced by RC in Phase 2, but lets Phase 1 be
validated on its own (you can *see* 3D shaded boxes/spheres).

### Phase 1 done = thousands of instanced box/sphere impostors render in 3D with
correct depth + normals, emitting a G-buffer, shown via stopgap lighting.

---

## Phase 2 status — IMPLEMENTED (screen-space RC)

`createRadianceCascades.ts` replaces the stopgap: fused raycast+merge cascade passes
(ping-pong `cascA`/`cascB`, top→down) + a final gather, all fullscreen render passes over
the G-buffer. Validated in `packages/debug/test.html` (lil-gui panel exposes the live
knobs). Key correctness fix: the occlusion test compares **world-space camera distance**
(probe vs sampled surface), NOT raw reverse-Z ndc — a constant ndc bias is unusable
because reverse-Z is non-linear over [near, far=20000], so it either does nothing or
skips every occluder. `zBias` is now a **world-unit** dead-zone (~2–8) that removes
self-occlusion acne while keeping contact shadows. Live params (uniform-driven):
`marchSteps`, `zBias` (world), `baseIntervalPx`, `ambient`, `ambientFill`, `sky`.

Known remaining (acceptable for the milestone): no light sources in the test scene → looks
like sky-lit AO rather than directional/colored GI (add a `LightEmitter` to see bleed);
single depth layer → silhouette light-leaks (world-space probe RC is the later upgrade);
mild banding at low cascade-0 ray counts.

## Phase 2 contract summary (for the separate RC design)

Phase 2 (screen-space Radiance Cascades) consumes: **depth**, **world normal**,
**albedo**, **emission**. Marches the depth buffer for visibility. Replaces the
stopgap composite. World-space probe-grid RC remains a possible later upgrade and would
reuse the same G-buffer (it additionally needs a global scene SDF/voxelization — out of
scope here).

## Status / open items
- Phase 0 + Phase 1 implemented; validated in the standalone scene
  `packages/debug/test.html` (`src/testScene/main.ts`) — box/sphere impostors, perspective
  reverse-Z, orbital camera, stopgap Lambert lighting.
- Normal encoding: currently full `rgba16f` (octahedral `rg16f` is a later memory win).
- Rounded boxes: back via IQ-style sphere-trace (analytic AABB entry + relative eps).
  If a perfectly edge-on rounded face ever flickers, raise `ROUND_STEPS` or switch that
  primitive to an analytic rounded-box (slabs + edge cylinders + corner spheres).
- Engine still references the removed `ShapeKind.Circle` (3 sites) — left broken until
  the engine 3D migration; the renderer + test scene are clean.
