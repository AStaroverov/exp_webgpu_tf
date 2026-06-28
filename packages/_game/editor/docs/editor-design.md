# Scene Editor — Design & Task Description

> A simple, classic Unity-like scene editor as a **new prototype package** at
> `packages/_game/editor`, built on top of the existing `engine` package. This is a design /
> task-description document, not final code. It states **what to build** and the **key technical
> decisions**, deliberately leaning toward the smallest thing that works and stays ECS-idiomatic
> (`CLAUDE.md`: components are data, systems query them, no speculative abstraction, no comments on
> self-describing code).

---

## 1. Overview & goals

Build a classic in-browser scene editor over `engine`. The six prototype goals:

1. **A simple classic editor like Unity's** — a WebGPU viewport flanked by panels.
2. **A scene tree (hierarchy)** of the current scene.
3. **A palette** to choose what to add to the scene.
4. **Add a collider.**
5. **Render the collider** (visualize the physics shape).
6. **Drag a shape along its axes** (a translate gizmo).

Scope: a **prototype**. Flat scene list (no nesting), single-select, translate-only gizmo, box/sphere/ground
node kinds (the three `BodySpec` kinds the engine already supports). No undo, no save/load, no rotate/scale
gizmo, no multi-select in the MVP. Everything is additive: the editor is a new app that imports `engine` and
`renderer` by relative path; one small, contained change lands inside `engine` (a reposition op, §4).

Two of the six goals are **already satisfied by the engine's spawn factories**: in
`packages/engine/src/ECS/Entities/RigidShapes.ts`, every `createRigidBox`/`createRigidSphere`/`createGround`
builds the **render shape and the physics collider as the same entity, in one call**. So "add a collider" =
"spawn via a factory", and "render the collider" = the SDF shape already drawn at that eid. The genuinely new
work is the **UI shell**, **selection/tree as ECS state**, the **picking + translate gizmo**, and the
**pose round-trip** that makes a drag actually move the body (§4).

**The one goal that needs a real decision, not a hand-wave, is goal 6.** The palette's box and sphere spawn as
`bodyType: "dynamic"` (verified `RigidShapes.ts:42,60`); only ground is `fixed`. So the gizmo's primary
targets fall under gravity. §4 commits to a concrete answer (kinematic-during-drag) rather than deferring it.

---

## 2. Architecture

### Where it lives

A new runnable Vite app at `packages/_game/editor`, cloning the `engine` package layout. The `_game/` nesting
is novel (all current packages are flat `packages/*`); register it explicitly in the root
`package.json` `workspaces` array as `"packages/_game/editor"` (match the repo's per-package-entry style, not
a glob).

### Build wiring (non-negotiable, copy from `engine`)

- **`config.vite.ts`** — copy `packages/engine/config.vite.ts` verbatim. It is the source of truth for the
  COOP/COEP headers (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`
  on both `server.headers` and `preview.headers`), `optimizeDeps.exclude: ["@dimforge/rapier3d-simd"]`, and
  the `worker: { format: "es", plugins: [wasm(), topLevelAwait()] }` block. `createEngine` throws when
  `crossOriginIsolated === false`; there is **no single-thread fallback**.
- **`package.json`** — copy `engine`'s (rename to `editor`, change the dev `--port`). It already lists every
  leaf runtime dep: `bitecs`, `@dimforge/rapier3d-simd`, `lil-gui`, `vite`, `vite-plugin-wasm`,
  `vite-plugin-top-level-await`, `gl-matrix`, `@webgpu/types`, `typescript`. `engine`/`renderer` are imported
  by **relative path**, not as package deps.
- **`tsconfig.json`** — `{ "extends": "../../tsconfig.json" }`.
- **`index.html`** — copy `engine`'s, point the module script at `/src/main.ts`, replace the full-screen
  canvas with the panel layout below.
- **Relative-import depth (verified, not guessed):** from a file under `packages/_game/editor/src/` the
  imports resolve to **`../../../renderer/src/...`** and **`../../../engine/src/...`** — both **three** `../`
  (src → editor → _game → packages). I computed this with `os.path.relpath`, not by counting in my head; do
  the same `Phase 0` smoke-import before writing much code, because one miscount cascades.

### How it consumes `engine`

`engine`'s entire public surface is `createEngine({ canvas, width, height }): Promise<EngineDI>` returning
`{ width, height, world, tick(delta), destroy(), setRenderTarget(canvas) }`. The editor is **a fresh
`main.ts`** that calls `createEngine` directly — it does **not** import `demo.ts`. So:

- `const engine = await createEngine({ canvas })`, then a fresh rAF loop calling `engine.tick(delta)` with
  delta clamped to `16.6667`. `tick()` reads the worker pose bank → `LocalTransform`, composes transforms,
  and renders. (`demo.ts`'s programmatic auto-orbit lives in **its** loop and simply isn't written here — there
  is nothing to "disable"; the editor just doesn't add it. Camera moves only from input, §6.)
- Spawn via the `RigidShapes.ts` factories. Despawn via `removeEntity(world, eid)` (bitecs).
- Read components anywhere via `getEngineComponents(world)` (`RigidBodyState`, `RigidBodyRef`, plus all
  `RenderComponents`: `Shape`, `Color`, `LocalTransform`, `GlobalTransform`, …).
- Camera state lives module-side in `renderer/src/ECS/Systems/ResizeSystem.ts`
  (`cameraAzimuth/Elevation/Zoom`, `setCameraPosition`); the editor reads/writes these for orbit and for the
  picking math (§6).

### Panel / viewport layout

A CSS-grid shell around the WebGPU canvas: **left = scene tree**, **center = canvas viewport**,
**right = palette + inspector**. The canvas is one grid cell with `display:block`; the resize system reads
`canvas.offsetWidth/Height` live each frame (`ResizeSystem.ts:121-122`), so a CSS-sized canvas just works.

**UI tech split** (the `debug` package is listed in workspaces but is **absent on disk** — do not plan around
`createDebugGUI`):

- **Scene tree** → **plain DOM** (~40 lines): one `<div>` row per node, click → select, an inline text input
  for rename. lil-gui has no tree widget; nesting folders to fake a scene graph fights the tool.
- **Palette + inspector** → **lil-gui** (already used heavily in `demo.ts`, which is essentially a palette
  already). Palette = a folder of "Add Box / Add Sphere / Add Ground" buttons + spawn params.
  Inspector = a `params` object rebound to the selected entity's fields on selection change. **Pose fields
  must be read from `RigidBodyState.position.get(eid, i)` every frame, not bound once** — the worker owns
  pose and a freshly-spawned dynamic body is still falling, so a one-time bind shows stale spawn pose. Pose
  *edits* from the inspector go through `PendingMove` (§4), identical to the gizmo.

No new UI dependency.

---

## 3. Scene model

The scene **is** ECS state. The editor adds two marker components in its own package and one editor-only side
map; everything else is a query.

### Scene tree — `SceneNode` marker component

The world is **flat today**: there is no `Parent`/`Children` component; `createEngine` passes a `stubChildren`
that returns `0` children for every entity, so the transform system's second pass is a no-op. For the
prototype's flat list, **no transform parenting is needed** — only a flat membership set.

Add an editor-package component **`SceneNode`** (marker; optionally one `u8 nodeKind` field for palette
grouping/icons — skip if not needed yet). The hierarchy panel renders `query(world, [SceneNode])` each frame.

Rejected alternatives and why:
- *Query an existing component* (e.g. `[RigidBodyState]`) — couples the tree to physics and won't see future
  non-physical nodes (lights, empties). Leaky.
- *Editor-side `Set<eid>`* — an imperative duplicate of world state; `CLAUDE.md` explicitly rejects
  module-side buffers that should be world state. The query **is** the membership set ("the fastest search is
  the one you don't do").

Because `createEntityId` pulls from a **shared monotonic counter and eids never recycle**, an eid is a stable,
safe UI row key for the scene's lifetime — no ABA problem.

Nesting (`Parent`/`Children`) is deferred: the transform system already accepts a `Children`-like, so it can
be added later by replacing `stubChildren` — **but do not build it now**.

### Names / labels — editor-side `Map<eid, string>`, not a component column

bitecs columns store numbers only; there is no string storage, and a label means nothing to the worker (never
crosses the SAB). Names affect only the panel, never a query or a system → keep an editor-only
`Map<number, string>` (label by `nodeKind + counter` at spawn, edit via the tree's rename field). Drop the
entry on removal via `observe(world, onRemove(SceneNode), …)` (the pattern `RigidBodyState.ts:39` already
uses). Safe because eids never recycle — no stale-key ABA. This is UI state and legitimately lives
module-side, exactly like camera azimuth/zoom.

### Selection — `Selected` marker component

Selection is consumed by **systems** (gizmo, highlight), so it is world state, not a stray
`selectedEid` variable that every consumer would have to import and branch on (the anti-pattern `CLAUDE.md`
rejects). Existence-based:

- Click → editor clears the previous `Selected` (single-select), then `addComponent(world, hitEid, Selected)`.
- The **gizmo system** queries `[Selected, RigidBodyState]` → exactly one place decides what the gizmo
  attaches to.
- The **highlight system** queries `[Selected, Shape]` and tints (save `OriginalColor` on add, restore on
  `onRemove(Selected)`), mirroring the existing magnitude-tint pattern.

Hover, if added later, is a **separate** `Hovered` marker — don't overload `Selected` with a state enum.

### Palette / node types

Add-to-scene = call the existing factory at a default pose, then tag the returned eid with `SceneNode`
(+ default label). The three factories cover all three `BodySpec` kinds:

| Palette item | Factory | `BodySpec.kind` | Body type |
|---|---|---|---|
| Box | `createRigidBox` | `box` | **dynamic** |
| Sphere | `createRigidSphere` | `sphere` | **dynamic** |
| Ground | `createGround` | `groundBox` | fixed |

**No new factories** are required to ship the six goals. The dynamic body type of box/sphere is exactly why
the gizmo needs §4's kinematic-during-drag — it is not an afterthought.

---

## 4. Editing & the physics-pose conflict (the central decision)

### The conflict, precisely

The **worker owns pose**. Every fixed step it writes Rapier's translation/rotation into the back pose bank and
publishes; main only **reads** (`RigidBodyState.position.get` resolves the read bank per call;
`createApplyRigidBodyToTransformSystem` blits it into `LocalTransform.matrix` via
`mat4.fromRotationTranslation(_q, _t)` **every tick**, `:55`). The op channel has **only** `SPAWN_BODY` +
`DESPAWN_BODY` (`opChannel.ts:5-8`). Therefore any main-side write of a new position into `LocalTransform` or
the pose bank for a **bodied** entity is **overwritten on the next `tick()`** — by the live sim for dynamic
bodies, and by the next bank publish even for a settled one.

> **Important exception (load-bearing):** the apply system queries `[LocalTransform, RigidBodyState]`. An
> entity **without** `RigidBodyState` (e.g. a gizmo arm — §6) is never touched by it, so its `LocalTransform`
> is pure main-side state and main writes to it are **not** clobbered. The "main writes always get clobbered"
> rule is true **only for bodied entities**.

### Options considered

- **(a) Edit mode = pause physics globally.** The worker self-clocks (`setTimeout` loop,
  `physics.worker.ts:201`); pausing it needs a *new* control channel and doesn't, by itself, let you move a
  body. Heavy and incomplete. **Rejected.**
- **(c) Edit render transform + despawn/respawn on drop.** Both ops already exist (zero op-channel change).
  But it requires suppressing the apply-system for the dragged eid (a per-entity `if` in the hot apply loop),
  and respawn mints a **fresh eid** every drop — churning the tree/selection identity and the label map. Fine
  for a one-shot drop, terrible for a continuous drag. **Rejected as the primary path.**
- **(b) Add a `MOVE_BODY` reposition op.** The worker applies it with `body.setTranslation(t, true)` /
  `body.setRotation(q, true)`, then publishes the new pose back through the bank **exactly as today** — so
  main's read path shows the moved object with **zero changes to the read side**. This preserves the engine's
  one invariant (worker is sole pose authority, main only reads) instead of fighting it.

### Decision: **(b) `MOVE_BODY` op + a `PendingMove` command component, with kinematic-during-drag for dynamic bodies**

The leanest change that respects the codebase's invariant and is ECS-idiomatic (`CLAUDE.md`: command
components, not flags + branches; "spawn is a component-add" precedent). Concrete new pieces:

**1. `opChannel.ts` — extend the discriminated union, not just a number.** `OpCode` is an **object literal**
`{ SPAWN_BODY: 1, DESPAWN_BODY: 2 }` and the codec gates on `op.op === OpCode.X` over a
`StructuralOp = SpawnBodyOp | DespawnBodyOp` union (`:36,87,107`). Adding MOVE means **all** of:
- `MOVE_BODY: 3` on `OpCode`.
- a `MoveBodyOp = { op: typeof OpCode.MOVE_BODY; eid; x,y,z; qx,qy,qz,qw }` member **added to the
  `StructuralOp` union** (so `decodeOp`'s return type covers it and every consumer must narrow).
- a `moveBody(eid, pos, rot)` constructor, an `isMoveBody(op)` guard.
- branches in **both** `encodeOp` and `decodeOp`. Payload fits trivially: stride is
  `OPS_PAYLOAD_STRIDE = 11`, eid in slot 0, so position(3)+quaternion(4)=7 floats land in slots 1–7 (spawn
  already uses 0–9). **No `LAYOUT_VERSION` bump** — the op ring is opcode-gated, not column-keyed.

> **Note:** the existing `export type OpCode = (typeof OpCode)[number & keyof typeof OpCode]` alias resolves to
> `never` (`number & keyof {…}` is `never`); it compiles only because nothing consumes it. The real
> discriminant is the per-op `op` literal in the union — extend the **union**, and treat the alias as cosmetic
> (optionally fix it to `(typeof OpCode)[keyof typeof OpCode]` while you're there, but that's not required).

**2. Worker `drainOps` — the `else = despawn` landmine MUST be fixed first.** Today (`physics.worker.ts:121-125`):
```
const op = decodeOp(opcode, payload, slot);
if (isSpawnBody(op)) spawnBody(op);
else despawnBody(op.eid);          // <-- a MOVE op falls HERE and DELETES the body
```
A `MOVE_BODY` op decoded into this branch would be treated as a despawn and **silently destroy the entity on
the first drag**. The branch must become explicit:
```
if (isSpawnBody(op)) spawnBody(op);
else if (isMoveBody(op)) moveBody(op);
else despawnBody(op.eid);          // isDespawnBody
```
`moveBody(op)` resolves the pid via `RigidBodyRef.id[eid]` — the same lookup despawn uses — and **must apply
the same `pid !== 0` guard** (`:163-164`): the entity may have been despawned before its move op drains, and
`pw.getRigidBody(0)` is undefined behavior. Then `setTranslation({x,y,z}, true)` / `setRotation({x,y,z,w},
true)`. The step loop publishes the new pose back next frame, unchanged.

**3. `PendingMove` command component** (editor package) carrying `{ x, y, z, qx, qy, qz, qw }`, plus
`createApplyPendingMoveSystem` that queries `[PendingMove]`, producer-gates
`pushOp(encodeOp(moveBody(...)))`, then `removeComponent(world, eid, PendingMove)`. The gizmo writes intent
as **data** (`addComponent(world, eid, PendingMove, …)`), never an imperative op call at the drag site —
mirroring how `RigidBodyState.addComponent` **is** the spawn command. Producer-gating (`if (sab.isProducer)`)
prevents any echo, exactly as despawn does.

**4. Dynamic bodies during drag — the real answer, not a deferral.** Box and sphere are dynamic, so between
move ops gravity pulls them and `setTranslation` just fights it. So MOVE alone is **not** enough for goal 6 on
its primary targets. Two viable ways to make the drag actually hold:

- **(b1) Kinematic-position while grabbed (recommended).** Add a tiny **`setBodyType` op** in the same channel
  (`SET_BODY_TYPE: 4`, payload = eid + a type code). On drag-start the gizmo emits "→ kinematicPositionBased"
  for the grabbed eid; each drag-frame writes `PendingMove`; on drop it emits "→ dynamic". A
  kinematic-position body ignores gravity and honors `setNextKinematicTranslation` / `setTranslation`
  exactly, so the drag holds and the body resumes falling on release. This is one more opcode of the **same**
  shape as MOVE (union member + encode/decode + worker branch + a `PendingBodyType` command component) — no
  new channel, no pause loop. **This is the MVP path for dynamic drag.**
- **(b2) Spawn palette box/sphere as `fixed` in the editor.** Cheapest possible (zero extra op): the editor
  passes `bodyType: "fixed"` through a thin editor-side spawn wrapper (or a one-line factory variant) so
  every placed object stays put and is freely draggable; a future "Play" toggle would flip them to dynamic.
  Trade-off: you don't see live falling/settling in edit mode, which is arguably *correct* for an editor.

**DECISION (chosen): (b2) — the editor spawns palette box/sphere as `fixed`.** Absolute minimum (zero extra
op), every placed object stays put and is freely draggable, and it gives a clean edit/play split: a future
"Play" toggle flips them to dynamic. The trade-off (you don't see live falling/settling in edit mode) is
accepted — it is arguably correct for an editor. (b1) remains the documented upgrade path if "see it fall"
during editing later becomes a requirement.

Consequence for the plan: Phase 4 does **not** need the `SET_BODY_TYPE` op or `PendingBodyType` component — it
ships only `MOVE_BODY` + `PendingMove`. Goal 6 is **honestly closed** because every shape the palette spawns is
`fixed` and so the gizmo's `MOVE_BODY` holds with no gravity fighting it.

Edits are async (worker drains at its next phase boundary, then publishes; main sees the result ~1–2 frames
later) — the same model spawn already uses. For drag *feel*, see M-note below.

> **Drag responsiveness (do not route the visual through the worker round-trip).** Routing every drag frame
> through `PendingMove → worker → publish → read` adds ~1–2 frames of lag to the *visible* object and (if
> gizmo arms are entities) to the handles. For a smooth drag: each drag-frame **also** write the new pose
> *optimistically* on the main side for immediate feedback, and let the worker pose **reconcile** it. Concretely:
> the gizmo arms are render-only entities (no `RigidBodyState`), so writing their `LocalTransform` directly is
> never clobbered (see the exception above) — move them with the cursor every frame for instant feedback. The
> grabbed **body** is kinematic during the drag (b1), so its worker-published pose tracks the `PendingMove`
> targets within a frame or two and the small lag is invisible in practice. Validate the feel in Phase 4
> (inspector-slider drive) before building the gizmo on top.

---

## 5. Colliders

### What "add a collider" means here

There is **no standalone collider concept**. A collider is *implied by `BodySpec.kind`*; the worker turns the
kind into exactly one centered Rapier collider (cuboid / ball). So **"add a collider" = "spawn a body via a
factory"** (§3 palette), and the three supported colliders are box / groundBox / sphere. No
add-collider-to-existing-body path exists, and it is out of scope for the prototype.

### Collider vs render shape

They are **separate parallel descriptions of the same eid**, kept consistent by construction in the factories
(`RigidShapes.ts`: same `sx/sy/sz` → render `depth` + collider `halfExtents`, same center `z`):

- Render: the `Shape` component on main — box `values = [width, height, depth]` (**full** extents), sphere
  `values[0] = radius`.
- Collider: `BodySpec` dims live only in the worker's Rapier world and the SPAWN op payload. **Main has no
  component holding the collider dimensions after spawn.**

### Rendering the collider (goal 5)

The default-and-cheapest answer: **the SDF shape already drawn at the eid is the collider visualization** —
for box/sphere/ground the render shape and collider coincide by construction, and (in a "play"/dynamic mode)
an object falling under gravity already proves the collider is live. For the MVP this satisfies goal 5 with
**zero new code**.

If a visually distinct wireframe overlay is wanted (e.g. to reveal true rotation the impostor hides — the 2.5D
SDF only renders yaw + spheres faithfully, even though the transform matrix carries the full quaternion), add
it as an **edit-only overlay layer**, not an engine change:

1. **Source the collider dims on main** off the render `Shape.values` (box → halve `[w,h,d]`; sphere →
   `values[0]`) — already per-eid on main, valid while render == collider (true for the factories). No new
   component. Decoupled route (only if collider ≠ render later): a small main-side
   `ColliderShape { kind, hx, hy, hz | r }` written alongside `RigidBodyState.addComponent` in `RigidShapes.ts`.
2. **A wireframe-draw system** gated by a `ColliderDebugDraw` marker so it costs nothing when off. The
   renderer is solid SDF impostors with **no line primitive**, so this is a small new line-list pipeline
   bound to `viewProjMatrix`, injected after `renderFrame`: 12 box edges from center ± halfExtents transformed
   by the eid's `GlobalTransform.matrix` (full rotation, so the wireframe shows true orientation even when the
   impostor doesn't); a 3-ring gizmo for spheres.

Recommendation: ship the MVP with **the existing SDF shape as the collider viz**; build the wireframe overlay
only if the distinction is actually needed.

---

## 6. Picking & gizmo

### Camera model (from `ResizeSystem.ts`, verified)

True **orthographic**, tilted top-down, **reverse-Z**, single shared module-level camera. State:
`cameraPosition{x,y}` (look-at on the `z=0` ground plane), `cameraZoom.value` (**pixels per world unit** —
`halfH = h/2 / zoom`, clamped min `0.01`), `cameraElevation.value` (deg, clamped 1–89.9, default 70),
`cameraAzimuth.value` (deg, default 45). It exports `viewProjMatrix` and `cameraRayDir`
(= `normalize(target − eye)`; all rays parallel under ortho).

Matrices are built from **CSS** width/height (`canvas.offsetWidth/Height`), and pointer events are CSS px, so
**do all mouse math in CSS pixels** with `pxPerWorld = cameraZoom.value`.

### Screen ↔ world (the load-bearing math)

Build the orthonormal view basis from the angles (do **not** read it out of `viewProjMatrix` — that has the
reverse-Z `ndcFix` baked in). Verified against `ResizeSystem`'s `dir`/`lookAt`:

```
right = ( -sin az,          cos az,          0 )
upScr = ( -sin el·cos az,  -sin el·sin az,   cos el )
fwd   = cameraRayDir = ( -cos el·cos az, -cos el·sin az, -sin el )
```

**Screen px → world ray** (mouse `sx,sy` in CSS px relative to canvas, top-left origin):

```
dx = sx - cssW/2;  dy = cssH/2 - sy
offset    = right·(dx/zoom) + upScr·(dy/zoom)
target    = (cameraPosition.x, cameraPosition.y, 0)
rayOrigin = target + offset - fwd·D           // D = a back-off along the ray
rayDir    = fwd
```

`D` only slides the origin **along** a parallel ray, so it cannot change an analytic hit `t` — any value in
the depth range `[0.1, dist+200]` (= `[0.1, 300]`, `dist=100`) is fine; use `D = 200` and don't rely on a
magic number. (It *does* matter for the world→screen path below — keep that path using `viewProjMatrix`, which
clips correctly.)

**Ray → world point on plane `z = z0`:** `t = (z0 - rayOrigin.z) / fwd.z` (`fwd.z = -sin el`, never 0 for
`el ∈ (1, 89.9)`), `worldPoint = rayOrigin + fwd·t`.

**World point → screen px** (for handle labels): `clip = viewProjMatrix · vec4(world,1)`;
`sx = (clip.x/clip.w·0.5+0.5)·cssW`, `sy = (1 - (clip.y/clip.w·0.5+0.5))·cssH`.

### Picking — CPU analytic ray-vs-shape, against the FULL transform

There is **no GPU id/picking pass** today, and adding one is too much for the prototype. Use **CPU analytic
ray-vs-shape on main**, against the exact transform the renderer composes:

- Iterate `query(world, [GlobalTransform, Shape])`, **excluding `GizmoHandle`** (see below) so handles aren't
  pickable as scene objects, and (optionally) excluding gizmo arms by simply not querying them.
- **Rotation is the FULL quaternion, not yaw.** `LocalTransform.matrix` is built by
  `mat4.fromRotationTranslation(fullQuaternion, …)` (`createApplyRigidBodyToTransformSystem.ts:55`) — it bakes
  in all three rotation axes even though the SDF only *renders* yaw. So a box test that reconstructs a
  yaw-only frame would **disagree with the physics body** for any tumbled dynamic box (the common case in a
  play mode). Do **not** reconstruct rotZ. Instead:
  - **Box:** invert the eid's `GlobalTransform.matrix.getBatch(eid)` (full 4×4) into a scratch matrix, transform
    `rayOrigin` and `rayDir` into the box's local frame, then run a center-origin slab test against
    half-extents (`Shape.values[0,1,2] / 2`).
  - **Sphere:** ray-vs-sphere at the world-space center (translation column of the matrix), `values[0]=radius`
    — rotation-invariant, so no inverse needed.
- Keep nearest hit by `t`. **No allocation in the loop** — reuse scratch vec3s / a scratch mat4 (the
  `mat4.invert` writes into a reused matrix). Other shape kinds fall back to AABB until needed.

> If you adopt edit-mode (b2) where everything is `fixed` and upright, the full-matrix inverse degenerates to a
> yaw-only inverse anyway and the box test is exact. If you adopt (b1) and bodies tumble, the full-matrix
> inverse is what keeps the pick matching the body. Using the full matrix is correct in **both** cases, so
> always use it — it is not more code than reconstructing rotZ.

On hit → set `Selected` (§3). The DOM tree is the alternative selection path (and a fine fallback for the very
first slice if canvas picking isn't ready yet).

### Gizmo rendering

No existing line/overlay path. Two options:

- **(A) Gizmo handles ARE entities (recommended for the prototype).** Spawn three thin elongated boxes (via
  `createRectangle`) as X/Y/Z arms positioned at the selected entity, tagged `GizmoHandle{axis}`. Crucially:
  - **Do NOT tag them `SceneNode`** (else they pollute the hierarchy tree + label map) and **do NOT give them
    a label**.
  - **Do NOT give them `RigidBodyState`** — they are render-only, which is exactly why the apply-system never
    clobbers their `LocalTransform`, so the editor positions them every frame with a direct main-side write
    (instant, no worker round-trip). This is the clean path the §4 exception enables.
  - **Exclude `GizmoHandle` from the scene-object picker** (a `Without(GizmoHandle)` on the pick query), and
    run a **separate** handle-pick query *first* so grabbing a handle wins over selecting whatever's behind it.
  - Picking a handle uses the **same** CPU box ray test; they're depth-tested for free.
  - Cons: they live in the lit scene (VCT GI lights them) and a pure-Z arm is foreshortened under the tilted
    cam. Fine for a prototype.
- **(B) Dedicated overlay pass.** A `line-list` pipeline bound to `viewProjMatrix`, drawn into the present
  target with depth disabled or `greater-equal`. Crisp, unlit, never lit/foreshortened — but a whole pipeline
  + vertex buffer to maintain.

Go with **(A)** now; promote to (B) only if "lit gizmo" or Z-handle foreshortening becomes a real problem.

### Drag-to-translate math (zoom-aware, per-axis)

Given chosen world axis `A` (unit, e.g. `(1,0,0)`) and mouse delta `(Δsx, Δsy)` in CSS px (Y-down from the
event), project the axis onto the screen basis and divide the mouse delta projected onto it:

```
ax = dot(A, right)·zoom            // px per world-unit, screen X
ay = dot(A, upScr)·zoom            // px per world-unit, screen Y (screen-up)
mx =  Δsx;  my = -Δsy              // flip to screen-up
denom = ax*ax + ay*ay              // guard ~0 (axis edge-on)
worldDelta = (mx*ax + my*ay) / denom
newTranslation = oldTranslation + A·worldDelta
```

The `zoom²` in `denom` vs `zoom¹` in the numerator makes the result scale `1/zoom` automatically (zoomed in →
a px drag moves the object less in world space — correct). Recompute `right`/`upScr` at drag-start (they only
change on camera orbit). For `A=(0,0,1)` under a tilted cam, `ax≈0, ay=cos el·zoom` → a purely vertical-screen
drag, which feels right; clamp when `denom→0`.

**Write-back:** each drag-frame writes the visual immediately (gizmo arms' `LocalTransform` directly; the
grabbed body is kinematic so it tracks) **and** records the target as `PendingMove` (§4), which the apply
system emits as `MOVE_BODY`. The worker publishes the moved pose back as the authoritative reconcile. On drop:
flip the body back to dynamic (b1) via the `setBodyType` op, or leave it (b2).

### Camera vs drag input

There is no existing mouse-orbit handler (and the editor doesn't run `demo.ts`'s programmatic orbit). The
editor adds its own and picks a modifier convention: **plain drag = pick / move object (or gizmo handle);
Alt+drag (or middle-mouse) = orbit camera (`setCameraAzimuth`/`setCameraElevation`); wheel = zoom
(`setCameraZoom`)**.

---

## 7. Implementation plan (phased, MVP-first vertical slices)

Each phase is a runnable slice. Goals in parentheses.

**Phase 0 — App skeleton.** Create the package (`package.json`, `config.vite.ts`, `tsconfig.json`,
`index.html`, `src/main.ts`); register in root workspaces; **smoke-test the `../../../renderer` /
`../../../engine` imports resolve**, then verify `createEngine` boots under COOP/COEP with a fresh rAF `tick`
loop. *(goal 1)*

**Phase 1 — Palette + flat tree + collider (free).** lil-gui palette calls the three factories at a default
pose; on spawn, tag the eid with `SceneNode` and add a `Map<eid,name>` label. Plain-DOM tree renders
`query([SceneNode])`; cleanup label on `onRemove(SceneNode)`. The spawned shape **is** the rendered collider.
*(goals 2, 3, 4, 5)*

**Phase 2 — Selection + inspector.** `Selected` marker; tree click and (Phase 3) canvas pick set it
single-select; highlight system tints `[Selected, Shape]` (save/restore `OriginalColor`). lil-gui inspector
rebinds to the selection — **pose read per-frame from `RigidBodyState.position.get`** (not a one-time bind),
live color edit. *(goal 1 polish)*

**Phase 3 — CPU picking + camera input.** Implement the screen→world ray and ray-vs-box (full-matrix inverse)
/ ray-vs-sphere pick (§6); canvas click selects. Add Alt-drag orbit + wheel zoom. *(goal 6 prerequisite)*

**Phase 4 — Reposition op + `PendingMove`.** Land `MOVE_BODY` in `opChannel.ts` (union
member + encode/decode + guard), **fix the worker `drainOps` `else`-branch landmine** (explicit
`isMoveBody`/`isDespawnBody` branches + `pid !== 0` guard), the `PendingMove` command component, and
`createApplyPendingMoveSystem`. Dynamic-drag is handled by the chosen **(b2)** path — the editor's spawn
wrapper passes `bodyType: "fixed"`, so no `SET_BODY_TYPE` op / `PendingBodyType` component is needed here.
Round-trip a position edit end-to-end from **inspector sliders** first to validate latency/feel before the
gizmo. *(goal 6 core)*

**Phase 5 — Translate gizmo.** Spawn three `GizmoHandle{axis}` render-only entities at `[Selected,
RigidBodyState]` (no `SceneNode`, no `RigidBodyState`); a handle-pick query runs before the scene-object
picker; drag math (§6) writes the gizmo-arm `LocalTransform` directly (instant) + `PendingMove` (authoritative).
On grab/drop, emit the body-type flip (b1). *(goal 6 complete)*

**Later (not MVP):** collider wireframe overlay (§5); real `Parent`/`Children` hierarchy; rotate/scale gizmo
(the wireframe overlay is the way to *see* true rotation the impostor hides); multi-select; undo; scene
save/load (eids never recycle, so serialization is stable, but reload must respect the shared counter); a
dedicated overlay/line pass for crisp unlit gizmos.

---

## 8. Open questions / risks

1. **Dynamic-drag mechanism — RESOLVED: (b2).** The editor spawns palette box/sphere as `fixed`; objects stay
   put and are freely draggable, with a future "Play" toggle to flip them to dynamic. No `SET_BODY_TYPE` op in
   the MVP. (b1) kinematic-during-drag stays documented as the upgrade path if live falling in edit mode is
   ever wanted.
2. **`drainOps` else-branch.** Adding `MOVE_BODY` without changing the worker's binary
   `if (isSpawnBody) … else despawnBody()` would make the **first drag silently delete the entity**. Phase 4
   must split the branch explicitly and reuse the `pid !== 0` guard. Highest-risk concrete edit; called out so
   it can't be missed.
3. **Full-quaternion picking.** The transform matrix bakes the full rotation (not yaw). Box picking inverts
   the full `GlobalTransform.matrix`; this is correct whether bodies are upright (b2) or tumbled (b1), so
   always use the full inverse — never reconstruct rotZ.
4. **2.5D rotation is a half-truth.** Only yaw + spheres render rotation faithfully; a tumbling box looks
   upright while its collider and transform are fully rotated. A rotate gizmo would visually lie —
   deprioritized. The collider wireframe overlay (§5) is the way to *see* true rotation if needed.
5. **Drag latency vs visual feedback.** Routing the *visible* object solely through the async worker
   round-trip lags ~1–2 frames. Mitigated (§6) by writing the gizmo arms' `LocalTransform` directly each
   drag-frame (render-only entities, never clobbered) and keeping the grabbed body kinematic so its published
   pose tracks the targets. Validate feel with inspector sliders in Phase 4 before building the gizmo.
6. **Production COOP/COEP.** Any host serving a built editor must send the same headers or SAB fails **only in
   production**. Note for deploy.
7. **`_game/` nesting + import depth.** Verified `../../../renderer/src` / `../../../engine/src` (three `../`)
   via `os.path.relpath`; still smoke-test in Phase 0, and confirm root tooling (oxlint/oxfmt) handles the
   deeper nesting.
8. **Gizmo handles as lit scene entities (option A)** get VCT-lit and a foreshortened Z-arm; they must be
   excluded from the scene-object picker and never tagged `SceneNode`. Acceptable for the prototype; option B
   (overlay pass) is the upgrade path if it looks wrong.
9. **`ColliderShape` component — build it or not?** Skip for MVP (read dims off `Shape.values`). Add only if
   collider geometry is ever meant to diverge from the render shape.
10. **`OpCode` type alias is cosmetic** (`never` today). Extend the `StructuralOp` **union** for `MOVE_BODY`;
    the alias is not the discriminant. Optionally repair it (`(typeof OpCode)[keyof typeof OpCode]`) in passing.
