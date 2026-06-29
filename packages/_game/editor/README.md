# viewer

An entity viewer over the `engine` package: pick a procedurally-built entity from the
selector and inspect it in the 3D viewport. Entities are defined in code (no scene
editing) under `src/Entities/`.

## Run

```bash
npm run dev   # in this package; serves on port 3355 under COOP/COEP (SharedArrayBuffer)
```

## Controls

- **Entity selector** (left panel) — choose which entity to display. The choice is
  remembered across reloads via `localStorage` (`viewer.entity`).
- **Regenerate** — rebuild the current entity (procedural builders use randomness).
- **Drag** the viewport — orbit the camera. **Wheel** — zoom.

## Adding an entity

1. Write a builder `build(world: EngineWorld): number` that returns the **root** eid.
   Create a root via `adoptEntity(world, createEntityId(world))` + `addTransformComponents`
   + `Children.addComponent`, then build the parts as children (`Children.addChild(root, part)`)
   in coords LOCAL to the root — the transform system composes them. The viewer holds only
   the root and clears the whole subtree (`removeEntityTree`). For static decor no physics is needed.
2. Register it in `src/Entities/registry.ts`.

Hierarchy is real transform parenting: parts are children of the root, so the whole entity
moves/clears as one. (One level deep for now; see `engine` `Children` + `TransformSystem`.)

Current entities: **tree** (`src/Entities/tree.ts`) — a root pivot with a trunk box + a
cluster of canopy spheres as children, randomized per build.

## Animation authoring

Data-driven clips replace per-entity hand-coded callbacks: a **clip** is a duration + a
set of **tracks** (one per animated part), each a list of **keyframes** (`t`, translate,
rotation). Keyframe `t` is a **normalized phase `[0,1]`**, not seconds — `duration` is the
only seconds knob, so changing it stretches/squashes the whole clip without touching keys;
the player samples at `phase = (clock % duration) / duration`. The player
decomposes/recomposes the part's `LocalTransform` matrix (gl-matrix
`getRotation`/`getTranslation`/`getScaling` → `fromRotationTranslationScale`), lerps the
translate, slerps the (sign-canonicalized) rotation with `smoothstep` easing, and writes
straight into the live `getBatch(eid)` column view. Parts are addressed by **stable bone
names** the builder publishes (`EntityInstance.bones`, e.g. `armR`, `unit/armR`,
`sword/root`), resolved to a live eid at build time, so clips survive entity rebuilds /
eid churn. Code (`src/anim/`): `clip.ts` (track data model + player), `editclip.ts`
(the authoring model — whole-entity snapshot keys + `editToClip` conversion),
`registry.ts` (in-session clip store + player wiring), `pose.ts` (matrix ↔ translate/Euler
for the inspector). Built-in procedural animations (unit/swordsman) still run alongside
clips, subject to the **one-writer-per-bone-per-frame** rule — authored clips target only
non-`root` bones; the procedural closures keep the root (bob/spring) and death alpha-fade.

### Authoring model: snapshot records, not per-track timelines

The authoring pipeline edits an `EditClip`: a **name**, a **total duration** (seconds),
the list of animatable **bones**, and a list of **records**. Each **Snapshot** appends a
record — a whole-entity pose (every animatable bone's translate + rotation captured at
once) tagged with an editable **key** number and a **% time**. There is no key picker on
the button: it just adds the next record. Records that share a **key** collapse into one
keyframe, merging bone-wise in record order (later wins), so you can build a keyframe up
over several snapshots. `editToClip` groups records by key, then transposes them into the
per-bone tracks the player consumes (keyframe `t = pct/100`, clamped to `[0,1]`). There is **no
canvas gizmo** and **no timeline scrub**: you pose parts by typing numbers, and a record
is a button press.

### In-session only (NOT persisted)

Authored clips live in a module-level registry for the current page session. They are
**not** saved to `localStorage` or JSON and **do not survive a reload**. To keep an
authored clip, click **Log clip to console** and paste the printed JSON into the game.
The only built-in clips are the demo `armRaise*` clips registered at startup.

### Authoring loop (Animation = edit → pose → Snapshot → % + duration → log)

1. **Enter edit mode** — set the **Animation** selector to **edit**. The right **Animation
   pipeline** panel enables and the pose inspector becomes editable. No animation runs in
   edit mode; parts hold the values you pose.
2. **Pose a part** — click a row in the left **tree** to select a part, then type into the
   inspector: translate **x/y/z** and rotation as **Euler degrees** (rx/ry/rz). The
   viewport updates live.
3. **Snapshot** — click **Snapshot pose**: it captures every animatable bone's current
   transform and appends a record (no key to pick). The first snapshot creates the clip,
   bound to the current entity TYPE.
4. **Set key + timing** — each record row has an editable **key** and **% time**; records
   sharing a key merge into one keyframe (so 10 snapshots at key 0 = one pose built up
   bone-by-bone). Set the clip's **Duration (s)** and **Clip name** above. ✕ removes a
   record.
5. **Preview** — pick the clip by its **name** in the Animation selector (it is registered
   live as you edit) to watch it loop; switch back to **edit** to keep authoring.
6. **Log** — click **Log clip to console** to print `{ name, loop, duration, tracks }` as
   JSON for manual integration into the game.

### Manual browser checks

WebGPU can't run in CI — verify these in the browser (`npm run dev`):

- Viewer is unchanged for non-animation use: entity select, Regenerate, orbit-drag, wheel
  zoom, tree selection + highlight, component list all still work.
- The built-in `armRaise*` clip appears in the Animation selector for **unit** and
  **swordsman**; selecting it raises the right arm and loops smoothly (no rotation snap at
  the loop seam — confirms slerp + sign-canonicalization).
- Existing hand-coded unit/swordsman animations still play (coexist) and are not broken;
  the unit **death** animation still **fades alpha** (applyColor survived the split).
- Animation = **edit** enables the right pipeline panel + the inspector; any other value
  disables both. Selecting a part → inspector shows its translate + Euler rotation; editing
  **x/y/z** / **rx/ry/rz** moves/rotates it live; values round-trip sensibly.
- Pose parts → **Snapshot**, re-pose → **Snapshot**: two record rows appear with editable
  **key** + **%** fields (keys default 0, 1, …); the clip's **name** shows up in the
  Animation selector. Two records set to the same key collapse into one keyframe.
- Pick the authored clip in the selector: it plays/loops, interpolating between keys
  (translate lerp, rotation slerp, smoothstep ease) rather than snapping.
- Edit a key's **%** or the **Duration**: the playing clip retimes accordingly. ✕ on a key
  row removes it.
- **Log clip to console** prints valid JSON (`name`, `loop`, `duration`, `tracks`) for the
  authored clip.
- Switching the **Entity** while a clip is authored clears the pipeline (clips are bound to
  one entity TYPE); rebuilding/regenerating the same entity does not break its clip.
- Reload the page: authored clips are **gone** (in-session only) — expected, not a bug.
