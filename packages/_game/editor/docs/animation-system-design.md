# Design: Data-Driven Animation Clips + In-Editor Authoring for the Entity Viewer

> Scope: a DESIGN/PLANNING document for `packages/_game/editor` (the entity viewer built on
> the engine: bitecs world, 2.5D SDF renderer, `SceneNode` root + `Children`, transform DFS
> composing `LocalTransform → GlobalTransform`). This replaces the hand-coded per-entity
> animation callbacks with one reusable, data-driven **clip + player**, plus an in-editor
> **select → pose → capture → playback** workflow. Lean, prototype-grade, no speculative
> generality (per `CLAUDE.md`).

---

## 1. Overview & goals

### 1.1 What exists today

- **Animation contract** — `EntityInstance = { root, animations }` where
  `EntityAnimations = Record<string, (delta: number) => void>`
  (`packages/_game/editor/src/Entities/registry.ts:7-14`). An "animation" is just a named
  closure that mutates the world each frame.
- **Per-frame drive** — `main.ts` runs exactly ONE selected closure per frame, then
  `engine.tick(delta)` (which runs the transform DFS). `currentAnimations` is swapped wholesale
  on every `build()`; `selectedAnimation$` (an rxjs `BehaviorSubject`, `state.ts:18`) picks the
  active name; `NONE = "none"` runs nothing.
- **The bespoke logic to subsume** — two distinct hand-coded styles:
  - **`unit.ts`** (`buildAnimations`, `:124-182`): a `Pose` struct of 5 abstract scalars
    (`headTilt, height, bobW, roll, alpha`), named target poses `REST/MOVE/DEAD` (`:128-130`),
    an exponential `blendTo` (`a = 1 - exp(-delta*6)`, `:135-140`), and `applyPose` (`:142-171`)
    which **procedurally rebuilds** the root, body, head and both arm `LocalTransform` matrices
    from those scalars every frame, plus a `sin(clock*2)` body bob (`:145`) and a per-part
    `Color.set$(... pose.alpha)` fade (`:168-170`). Note the body carries a baked `rotateX(π/2)`
    (`:152`) and the head a `rotateX(headTilt)` (`:157`).
  - **`swordsman.ts`** (`applyStance`, `:109-145`): `Key = {at, v}` arrays (`:25-72`) feeding a
    `sampleKeys(keys, p)` sampler with `smoothstep` easing (`:74-85`), but these keys drive
    **abstract scalar channels** (`SLICE_X/Y/Z/PITCH/WRIST/STEP`, `LUNGE_*`) that are then
    *combined with the rest pose and scaled by a live, eased per-stance weight*
    (`sliceW/lungeW`, `:110-112`, `:125-134`) and **written to two targets from one phase**: the
    arm matrix (`armMatrix`, `:136-139`) and the root (`rootMatrix` yaw + Y-step, `:141-144`).
    The sword child entity is parented onto the unit's exposed `hand` (`:99`).
- **Transforms are mat4-only.** `LocalTransform/GlobalTransform = NestedArray.f32(16)` — one mat4
  per eid (`packages/renderer/src/ECS/Components/Transform.ts:6-16`). There is **no stored
  quaternion**. `LocalTransform.matrix.getBatch(eid)` returns a **live `subarray` view** into the
  column (`packages/common/src/typedArray.ts:91`) — writing through it mutates the component in
  place; this is the write primitive `unit.ts`/`swordsman.ts` already use (`unit.ts:143`,
  `swordsman.ts:101-102`). The whole-matrix scalar API is **`setBatch(batchIndex, values)`**
  (`typedArray.ts:86`); `set(batchIndex, index, value)` is a **single-scalar** setter
  (`:76`) — see §3.2. `gl-matrix` 3.4.3 (`mat4/quat/vec3`) is a dependency and provides every
  function used below (`getRotation/getScaling/getTranslation/fromRotationTranslationScale/slerp/fromEuler`).
- **Full 3D rotation now renders** (prior SDF-shader fix uses the full per-instance rotation, not
  just yaw) — so authoring arbitrary quaternion orientations is faithful.

### 1.2 Goal

Replace the two bespoke styles with **one data-driven player** that consumes a serializable
**clip** (duration + per-part keyframe tracks of translate + rotation), and add an in-editor
workflow to author those clips: select a part, pose its translate + rotation via a gizmo /
inspector, scrub a timeline, and **capture** the posed transform into keyframes. Authored clips
appear in the same `selectedAnimation$` dropdown and play through the unchanged per-frame surface.

### 1.3 Prototype scope (what is in / out)

**In (the smallest slice that delivers select → pose → capture → playback):**

1. `Clip / Track / Keyframe` data model + a `makeClipPlayer(world, clip, resolveEid, restScale) → (delta, t)=>void`.
2. A **stable part identifier** so clips survive eid churn (§2 — the central decision).
3. Single-clip playback with loop / one-shot, lerp-pos + slerp-rot, `smoothstep` easing,
   and quaternion sign-canonicalization (§3.3).
4. Authoring: a translate gizmo + a single-axis screen-facing rotate ring, and a numeric inspector.
5. A timeline (scrub, play/pause/loop, capture, edit/delete keys), edit-mode vs playback split.
6. Serialization to `localStorage`, keyed by entity TYPE + clip name, surfaced in the selector.

**Out (deferred, but the format must not preclude them):**

- **Cross-clip / multi-stance blending** (the swordsman `sliceW/lungeW` ease-in/out mix). v1 ships
  single-clip playback; blending is a clean follow-up because the player separates *sample* from
  *write* (§3.5). The ease-in/out of the swordsman stances is **lost until this lands** (§1.4 H-note).
- **Animated scale / color tracks.** Scale is a build-time constant today; the alpha fade stays code.
- **Full 3-ring rotate gizmo.** v1 rotates only about the screen-facing axis where the screen-angle
  math is well-conditioned (§4.5); the two edge-on rings are deferred.

### 1.4 What replaces the hand-coded logic — and what stays code

| Hand-coded thing | Disposition |
|---|---|
| swordsman slice/lunge **scalar-channel arm + root motion** (`swordsman.ts:25-145`) | **Partially migrate.** The fully-*blended* pose (at `sliceW=1`) bakes into **two tracks** (`unit/armR` + `unit/root`). The **per-stance ease-in/out weight is NOT a transform track and is LOST in v1** — it returns only with the §3.5 blender. **Not lossless.** |
| unit `idle/movement/death` **discrete target poses** (`unit.ts:128-130`) | **Expressible as clips** (each pose = one keyframe; a transition = a 2-key clip). Migrate opportunistically. |
| unit **continuous body bob** `sin(clock*2)` (`unit.ts:145`) | **Keep as code.** Procedural sine, not keyframes; faking it needs many keys and is still wrong. |
| **exponential blend-to-target** `1-exp(-delta*6)` (`unit.ts:135`) and stance ease `1-exp(-delta*8)` (`swordsman.ts:110`) | **Keep as code.** A *state-driven spring* toward a live target, not a *time-driven* timeline. Fundamental impedance mismatch with a fixed-duration clip. |
| per-part **`Color.alpha` fade** (`unit.ts:168-170`, inside `applyPose`) | **Keep as code, but must be EXTRACTED first.** The fade lives *inside* `applyPose`, the same function a clip replaces. Split `applyPose` into a transform half and an `applyColor()` half before deleting the transform code, else the death-fade dies with it (§7 Phase 4). |

**Coexistence has a hard rule: one writer per bone per frame.** Both a `makeClipPlayer` closure and
a hand-written procedural closure are `(delta)=>void`; `main.ts` cannot tell them apart. They may
run side-by-side in the same `animations` Record **only if they write disjoint bones**. They do
NOT today: `unit.ts` `applyPose` rebuilds the **root** every frame (`:142-147`), and `swordsman.ts`
composes `unit.animations.idle` (`:156,160`) *plus* writes the **root** in `applyStance` (`:141-144`)
— that is already a double-write, last-writer-wins. A clip with a `root` track stomping a still-running
procedural `idle` is the same bug. **Decision (see §3.6):** a clip and a procedural closure must
never target the same bone in the same frame. For the swordsman specifically, clips target
`unit/armR` + `sword/*` only; the **root stays procedural** (the bob/roll spring is exactly what
clips can't express). The unified player owns everything keyframeable; genuinely
procedural-continuous motion stays as small code (per `CLAUDE.md`).

> Side note: the swordsman `rootMatrix` write is itself a latent bug — `mat4.rotateZ(rootMatrix,
> rootMatrix, …)` and `rootMatrix[13] += …` (`:141-142`) **mutate the root in place every frame**,
> accumulating yaw/translation because nothing resets the root first (unlike `unit.applyPose` which
> `mat4.identity`s first). The clip model resets-from-keyframes each frame, so migrating root motion
> to a clip would *fix* this; but per the one-writer rule we keep root procedural in v1, so this bug
> is out of scope (flag it).

---

## 2. Clip data model

### 2.1 The central problem: clips cannot key tracks by raw eid

Every `build()` (entity switch / scale change / regenerate) tears down and re-mints the **whole**
tree; `createEntityId` is monotonic and **never recycled**. A clip keyed by raw eid dies on the
first rebuild. A clip must key each track by a **stable identifier** resolved to a live eid at
author time AND at play time. This is the load-bearing data-model decision.

### 2.2 Decision: **named bones** (a `bones: Record<string, number>` published by the builder)

| Option | Verdict |
|---|---|
| **Child PATH/index** `[2]`, `[2,0]` | Robust to eid churn, needs no builder change, but **brittle to structural edits** — insert a part and every later index silently re-targets the wrong track. Opaque in JSON (`"path":[2,0]` is meaningless). |
| **Named bones** map `name→eid` from the builder — **CHOSEN** | Stable across rebuilds (names are source constants), **survives structural edits** (names don't shift on reorder), readable clips (`"track":"armR"`), and **extends the handle the builder already exposes** (`UnitInstance.hand`, `unit.ts:20,24`). |
| Bone as an ECS component `Bone{nameId}` | Over-engineered for a prototype (name-intern table, no queryable consumer). Rejected per `CLAUDE.md` "no speculative generality". |

**Why named bones win over paths here:** `buildUnit` already hands out a named part handle
(`hand: parts.armR`, `unit.ts:24`); a `bones` map is the natural generalization of that, not a new
abstraction. Paths would force a `lib/path.ts` with `eidToPath`/`pathToEid` DFS helpers and a
documented fragility; the bones map is strictly less code and strictly more robust. Cost: each
builder publishes its parts — a few lines, all parts already in scope.

### 2.3 Builder change — publish a flat `bones` map

Extend `EntityInstance` in `registry.ts`:

```ts
export type EntityInstance = {
  root: number;
  bones: Record<string, number>;
  animations: EntityAnimations;
};
```

`buildUnit` publishes its parts. **Two-step migration** for the `hand` field (which `swordsman`
reads directly today):

1. Add `bones`, **keep `hand`** so `buildSwordsman` still compiles unchanged:
   ```ts
   // unit.ts buildUnit
   return {
     root: parts.root,
     bones: { root: parts.root, body: parts.body, armL: parts.armL, armR: parts.armR, head: parts.head },
     animations: buildAnimations(world, parts),
     hand: parts.armR, // alias, removed in step 2
   };
   ```
2. Rewrite `buildSwordsman` to use `unit.bones.armR` instead of `unit.hand` (`swordsman.ts:99,102`),
   then drop `hand` from `UnitInstance`.

**Swordsman nesting → namespaced bone keys.** `buildSwordsman` composes a unit + a sword, parents
the sword under the unit's right arm, and currently returns `{ root: unit.root, animations }` with
**no bones exposed at the swordsman level** (`:164`). It must merge children's bone maps under
prefixes so a track can target any depth without knowing the tree shape:

```ts
const unit  = parts.unit(world, { scale });
const sword = parts.sword(world, { scale: SWORD_REL });
Children.addChild(unit.bones.armR, sword.root);

const bones = { ...prefix("unit/", unit.bones), ...prefix("sword/", sword.bones) };
return { root: unit.root, bones, animations };
```

`prefix(p, m) = Object.fromEntries(Object.entries(m).map(([k, v]) => [p + k, v]))` (cold path; FP
map is fine per the project's no-`forEach` rule). A track references `"unit/armR"` or `"sword/root"`
— both resolve through one flat map regardless of nesting. Writing the arm's `LocalTransform`
automatically carries the sword through the DFS; do **not** also write the sword's global transform
(double-applies).

### 2.4 Clip / Track / Keyframe (the serializable shape)

```ts
type Vec3 = [number, number, number];
type Quat = [number, number, number, number]; // gl-matrix order [x,y,z,w]

type Keyframe = {
  time: number; // seconds, absolute within the clip
  pos: Vec3;
  rot: Quat;    // sign-canonicalized vs the previous key at capture/load (§3.3)
};

type Track = {
  bone: string;     // a key into the instance bones map ("armR", "unit/armR", …)
  keys: Keyframe[]; // sorted ascending by time
};

type Clip = {
  name: string;
  entityId: string; // EntityDef.id ("unit"/"swordsman"/…), binds the clip to the entity TYPE
  duration: number;
  loop: boolean;
  tracks: Track[];
};
```

This is exactly the user's spec: a duration; the set of animated parts (one `Track` per bone); per
part a set of keyframes by time, each a transform (translate + rotation). `entityId` is the
`EntityDef.id` (`registry.ts:22-32`) — the clip is bound to the entity TYPE, not an eid.

**Scale: omitted from `Keyframe` in v1 — but the player must preserve each bone's build-time
scale.** No current animation touches scale; it is a build-time constant baked into the root once
(`unit.ts:47`, and the sword's `SWORD_REL`, `swordsman.ts:98`). **Decision: snapshot each track's
rest-scale once, in `build()`, and pass it to the player** (rather than storing scale per keyframe,
which would bloat keys and bake the current `selectedScale$` into the data):

- Rest-scale-at-build keeps the clip **scale-independent** (the same clip plays correctly at any
  viewer scale) and the format lean.
- `restScale` is captured **immediately after `build()`**, before any clip has run, from each bone's
  live matrix via `mat4.getScaling` — so it is genuinely the rest pose, not a value another clip
  left behind (§3.2 M-note). The player receives `restScale: vec3[]` parallel to its tracks and
  feeds it to the recompose.
- If a future clip genuinely needs animated scale, add an optional `scale?: Vec3` to `Keyframe` and
  resolve `key.scale ?? restScale` **once per key at load**, never as a per-frame branch on absence
  (per `CLAUDE.md`: no nullable field you null-check in a hot loop).

This lets the root *be* a track (its translate/rotation animate) while its build-time scale is
restored from `restScale`, not from a keyframe — no root-scale double-apply.

### 2.5 Capture (author-time → keyframe) and resolution

- **Capture** decomposes the posed live local matrix at the timeline cursor:
  ```ts
  const m = LocalTransform.matrix.getBatch(eid);
  mat4.getTranslation(pos, m);
  mat4.getRotation(rot, m);
  ```
  `mat4.getRotation` divides out scale internally, so `rot` is clean given uniform positive scale
  (true for all current parts — see §8.1). Capture writes parent-relative **LocalTransform** —
  exactly what the player replays and the gizmo edits; no global↔local conversion. Capturing the
  unit `body`/`head` faithfully yields their baked `rotateX(π/2)` / `rotateX(headTilt)` quaternions
  (`unit.ts:152,157`).
- **Quaternion sign-canonicalization (required, not optional, §3.3):** on capture, before upsert,
  flip the new key's `rot` sign if `quat.dot(prevKey.rot, rot) < 0`, so neighboring keys live on the
  same hemisphere. Same pass runs once at clip *load*. Without it, multi-key rotation tracks
  (the slice pitch has 4 keys) snap between segments.
- **Author-time bone lookup:** selection is an eid (`selectedEid$`). Invert the live `bones` map
  once per build into `eidToBone: Map<number, string>`. Note the swordsman merge can map two names
  to one eid (`unit/root` and the swordsman root are the **same** eid — `buildSwordsman` returns
  `root: unit.root`); the inverse drops one name, which is harmless (either name resolves back to
  the same eid). An eid with no published bone is not animatable — the "add track" / capture control
  disables for it ("publish what you want to animate").
- **Play-time resolution:** `resolveEid = (bone) => instance.bones[bone] ?? -1`, resolved **once**
  at player construction into `trackEid: number[]`. Each rebuild constructs a new player from the
  new `bones` map, so eid churn never reaches the clip. A track whose bone is absent (clip authored
  against a different entity, or a renamed bone) resolves to `-1` and is **skipped** — verified eid
  or explicit absence, never a silent wrong-eid write (per `CLAUDE.md`).

---

## 3. Animation player

### 3.1 Where it lives & how it plugs in

New file `packages/_game/editor/src/Animation/clip.ts`: the `Clip/Track/Keyframe` types,
`makeClipPlayer`, `sampleTrack`, and the `canonicalizeTrack` sign pass. No engine/renderer changes —
the player writes `LocalTransform` exactly as today's callbacks do. It plugs into the **unchanged**
`animations` Record: at build time, given clips loaded for the entity type and the per-bone
`restScale` snapshot taken right after `build()`,

```ts
const animations: EntityAnimations = { ...proceduralAnimations };
for (const clip of clipsForEntity) {
  const trackEid = clip.tracks.map((t) => bones[t.bone] ?? -1);
  const restScale = trackEid.map((eid) =>
    eid < 0 ? vec3.fromValues(1, 1, 1)
            : mat4.getScaling(vec3.create(), LocalTransform.matrix.getBatch(eid)));
  animations[clip.name] = makeClipPlayer(world, clip, trackEid, restScale);
}
```

`main.ts`'s per-frame line that runs `currentAnimations[selectedAnimation$.value](delta)` is
untouched for built-in procedural clips; authored clips are driven off `currentTime$` instead of
raw `delta` (so they are scrub/pause-able — §5.3).

### 3.2 Player construction & per-frame write

`restScale` and `trackEid` are computed by the caller in `build()` (§3.1) — pinned to the rest pose
*before any clip runs* — and passed in, so the player has no order-dependency on live state:

```ts
export function makeClipPlayer(world, clip, trackEid: number[], restScale: vec3[]) {
  const { LocalTransform } = getEngineComponents(world);
  const pos = vec3.create(), rot = quat.create();

  return (_delta: number, t: number) => {
    for (let i = 0; i < clip.tracks.length; i++) {
      const eid = trackEid[i];
      if (eid < 0) continue;
      sampleTrack(clip.tracks[i], t, pos, rot);
      // write straight into the live column view — zero copy, matches unit.ts/swordsman.ts
      mat4.fromRotationTranslationScale(LocalTransform.matrix.getBatch(eid), rot, pos, restScale[i]);
    }
  };
}
```

- **API correctness (H1):** the matrix is written via `mat4.fromRotationTranslationScale` directly
  into the **live `getBatch(eid)` view**. The earlier-drafted `LocalTransform.matrix.set(eid, m)` is
  a bug — `set(batchIndex, index, value)` (`typedArray.ts:76`) is a single-scalar setter and would
  write `undefined`. (If you ever need a copy-in path instead, the correct call is
  `setBatch(eid, m)`, `:86` — but writing into the view is leaner and what the existing code does.)
- All scratch (`pos/rot`) is allocated **once** outside the per-frame loop; the loop is a plain
  monomorphic `for` reading typed-array columns by eid — no `forEach`, no per-element closures, no
  allocation (per `CLAUDE.md` hot-loop rules).
- The write target is **`LocalTransform` only**; the `TransformSystem` DFS recomposes `GlobalTransform`
  **unconditionally** every `engine.tick` (no dirty-flag gating), and the tick runs *after* the
  closure in `main.ts`'s loop — so no one-frame lag and no change-mark needed.
- The player is driven by a **caller-supplied `t`** (the timeline cursor, §5), not its own `clock`,
  so scrubbing and pause come for free.

### 3.3 Per-track sampling + sign-canonicalization (generalizes `sampleKeys`)

```ts
// run ONCE at capture-upsert and at clip load — neighboring keys share a hemisphere
function canonicalizeTrack(track: Track) {
  const keys = track.keys;
  for (let i = 1; i < keys.length; i++) {
    if (quat.dot(keys[i - 1].rot, keys[i].rot) < 0) {
      const r = keys[i].rot;
      r[0] = -r[0]; r[1] = -r[1]; r[2] = -r[2]; r[3] = -r[3];
    }
  }
}

function sampleTrack(track, t, outPos, outRot) {
  const keys = track.keys;
  let a = keys[0], b = keys[0];
  for (let i = 0; i < keys.length - 1; i++) {
    if (t <= keys[i + 1].time) { a = keys[i]; b = keys[i + 1]; break; }
    a = b = keys[i + 1];
  }
  const span = b.time - a.time;
  const u = span > 0 ? smoothstep((t - a.time) / span) : 0;
  vec3.lerp(outPos, a.pos, b.pos, u);
  quat.slerp(outRot, a.rot, b.rot, u);
}
```

- **Interpolation:** `vec3.lerp` for translate, `quat.slerp` for rotation (the reason rotation is
  captured as a quaternion, not Euler — slerp gives correct shortest-path orientation interpolation;
  Euler would gimbal/wrap badly).
- **Why canonicalization is mandatory (H3):** `quat.slerp` only picks the short arc between *its two
  arguments*. `mat4.getRotation` does not guarantee hemisphere continuity across separately-captured
  keys, so a ≥3-key track (the slice pitch has 4, `swordsman.ts:40-45`) can take the short way on
  A→B and visually snap on B→C. `canonicalizeTrack` flips each key relative to its predecessor once,
  making every adjacent pair short-path consistent. This is what makes the smooth-loop claim (§3.4)
  reliable.
- **Easing:** reuse `smoothstep` (`swordsman.ts:74`) per segment — preserves the current feel for
  free. Promote to a per-clip/per-track curve enum only on a real second curve need.
- **Out-of-range / single-key:** mirror `sampleKeys` (`swordsman.ts:78-85`) — hold first before the
  first key, hold last after the last, constant for one key.

### 3.4 Loop vs one-shot

The `Clip.loop` flag selects the cursor wrap (done by the caller advancing `t`, §5.3):
`t = loop ? clock % duration : min(clock, duration)`. For a clean loop the author places a key at
both `time = 0` and `time = duration`; with canonicalization the slerp seam is continuous. The
timeline surfaces the duration marker so this is obvious.

### 3.5 Cross-clip blending — deferred, format-ready

The swordsman blends rest → slice → lunge with eased weights (`swordsman.ts:110-145`). Expressed
correctly, this is a thin blender over N single-clip players, not logic baked into one clip: sample
each layer into scratch, `vec3.lerp`/`quat.slerp` by an eased weight, write once. **Out of v1** —
and so the swordsman stances' ease-in/out is genuinely absent until this ships (§1.4). The format
already supports it (a stance = one `Clip`), so capture/playback bake in no single-clip assumption.
Add `makeBlendedPlayer` only when a real two-stance entity needs it (3rd-use rule).

### 3.6 One writer per bone per frame (the coexistence contract)

Because the player and procedural closures all write the same `LocalTransform` column and `main.ts`
runs exactly one named animation per frame, the only real hazard is a **composed** animation that
internally calls both (the swordsman calls `unit.animations.idle` then `applyStance`). The rule:

- **A bone is owned by exactly one writer in a given frame.** Either a clip track writes it, or a
  procedural closure does — never both.
- For the migrated swordsman: clips own `unit/armR` + `sword/*`; the **root stays procedural**
  (bob/roll spring). A swordsman clip therefore must NOT declare a `root` / `unit/root` track.
- This is enforced by convention + a one-line dev assert in the player setup (warn if a clip track
  bone is also touched by a still-registered procedural closure for the same entity). No runtime
  arbitration — the design forbids the overlap rather than resolving it.

---

## 4. Authoring — selection + transform gizmo

### 4.1 What exists / what is missing

- **Selection** = `selectedEid$` (in-session `BehaviorSubject<number>`, `state.ts:20`) + a DOM tree
  built by a `Children` DFS walk (`main.ts` `appendNode`, rows stamped `data-eid`); tree click →
  `selectedEid$.next(eid)`. Reaction is cosmetic only (highlight + read-only component list). The
  scrapped editor's translate gizmo + picking are **documentation only** in
  `packages/_game/editor/docs/editor-design.md` §6 — none of it is in `main.ts`.
- **Canvas pointer input is fully consumed by camera orbit** (`down$`/`move$`/`up$` →
  `setCameraAzimuth/Elevation`; wheel → zoom). Authoring needs an **input-mode split**: plain drag =
  manipulate, modified drag (or an explicit "orbit" modifier) = orbit.
- **No quaternion stored** — rotation is baked in the mat4 (§1.1). Capture/inspector/gizmo go through
  full mat4↔TRS decompose/recompose, NOT the scale-preserving single-axis `setMatrixRotate*`
  helpers (those compose only one axis and can't represent an arbitrary posed quaternion).

### 4.2 Reuse from `editor-design.md` §6 (verified, transfers unchanged)

The translate math is concrete and verified (`editor-design.md:350-405`), same `ResizeSystem` camera,
same CSS-px events: the view basis `right/upScr/fwd` from `cameraAzimuth/cameraElevation` (do **not**
read it back out of `viewProjMatrix` — reverse-Z `ndcFix` is baked in there), screen→world ray, CPU
ray-vs-box/sphere picking, and the zoom-aware per-axis drag formula
`worldDelta = (Δsx·ax + (-Δsy)·ay) / (ax² + ay²)`, `ax = dot(A,right)·zoom`,
`ay = dot(A,upScr)·zoom`.

**Crucial simplification vs the scrapped editor:** parts here are pure render entities under
`SceneNode`/`Children` with **no `RigidBodyState`** — so there is **no worker round-trip and no
`PendingMove`/`MOVE_BODY`** machinery. The gizmo writes the part's `LocalTransform` **directly**
through the live `getBatch` view; the DFS recomposes `GlobalTransform` on the next `engine.tick`.
The whole authoring path is main-side and synchronous.

### 4.3 Picking + handles

- **Canvas pick:** add a modifier branch to the existing `down$`. On a manipulate-click, build the
  screen→world ray and CPU-test against shapes (reuse §6's ray-vs-box/sphere) to set
  `selectedEid$`. Picking is additive to the tree (the tree still works).
- **Gizmo handles = render-only entities** (option A from §6): thin elongated boxes via
  `createRectangle` (already imported in `main.ts`), tagged with an editor marker
  `GizmoHandle{axis: u8}`, parented under `sceneRoot` — **NOT** under the selected entity and **NOT**
  via `Children.addChild` to anything the tree walk renders (they must not pollute the DFS tree UI).
  Repositioned each frame at the selected part's world translation
  (`getMatrixTranslation(GlobalTransform.getBatch(eid))`). A handle-pick query runs **first** so
  grabbing a handle beats selecting what is behind it.

### 4.4 Translate (write the part's local matrix)

On axis-handle drag, apply the §6 world-delta to the part's local translation by writing the live
view:

```ts
const m = LocalTransform.matrix.getBatch(eid);
applyMatrixTranslate(m, dx, dy, dz);
```

### 4.5 Rotate — screen-facing-axis ring → quaternion (constrained for the prototype)

Full 3D rotation now renders, so the 2.5D objection in §6/§8.4 is void. But the screen-angle scheme
is **only well-conditioned for the ring whose axis points toward the camera**; the two edge-on rings
have real degeneracies. **Decision: v1 rotates about ONE axis — the most screen-facing world axis**
(`A = argmax |dot(A, fwd)|` over X/Y/Z, chosen at drag-start). The other two axes are reached by
re-orbiting the camera so a different axis faces the screen. This is the leanest scheme that *works*;
the full 3-ring gizmo is deferred.

Per drag on the chosen axis `A`:

1. Drag-start: record the gizmo world center `C` (= part world translation); project `C` to screen
   px via `viewProjMatrix` → `Cs`. Record the start screen angle.
2. Per move: screen vector `v = (sx - Cs.x, sy - Cs.y)`. **Guard:** if `|v| < ε` (cursor at/near
   center, undefined angle) ignore the sample. Signed screen angle `θ = atan2(v.y, v.x)`.
3. Angle delta = `θ_now − θ_prev` (unwrap across ±π).
4. Sign correction: negate the delta if `dot(A, fwd) < 0` (ring seen from behind rotates visually
   backwards), resolved once at drag-start. Because `A` is the screen-facing axis, `|dot(A,fwd)|` is
   near 1 — the magnitude is well-conditioned (the edge-on ill-conditioning that broke off-axis rings
   does not arise).
5. Apply as a **world-axis pre-multiply** to the local rotation:
   ```ts
   const m = LocalTransform.matrix.getBatch(eid);
   mat4.getRotation(qCur, m); mat4.getScaling(s, m); getMatrixTranslation(/*out*/ tVec, m);
   quat.setAxisAngle(qDelta, A, angleDelta);
   quat.multiply(qCur, qDelta, qCur); // pre-multiply = rotate about WORLD axis
   mat4.fromRotationTranslationScale(m, qCur, tVec, s);
   ```
   Pre-multiply rotates about the **world** axis (intuitive for a fixed screen-space ring);
   post-multiply would be local-axis. All scratch (`qCur/qDelta/s/tVec`) is module-level reused — no
   per-frame alloc.

**Ring visuals:** there is no line primitive. For the prototype, approximate the single active ring
with N thin `createRectangle` segment-boxes tagged `GizmoHandle{axis}` (cheap, pickable by the same
box test). It is acceptable to **ship the inspector (§4.6) first** to validate the quaternion math
before any ring visuals exist — that unblocks pose→capture immediately.

### 4.6 Inspector — translate XYZ editable; rotation display-only readout

Replace the read-only `renderComponents` list with editable translate fields plus a **read-only**
rotation readout, driven by the same `selectedEid$.subscribe`, plain DOM + `fromEvent` (matching the
`scaleEl` input pattern).

- **Translate** is fully editable (a 3-vector has no aliasing problem): on `input`, write
  `applyMatrixTranslate` deltas or rebuild T while preserving R/S.
- **Rotation is a DISPLAY-ONLY Euler readout, NOT an editor (M1).** `quat → Euler` is not unique
  (gimbal, ±180 aliasing): a clean posed quaternion can display as ugly Euler and typing it back
  `fromEuler`s a *different* quaternion than was captured. So rotation **authoring goes through the
  gizmo** (§4.5), which composes incremental quats and never round-trips through Euler. The matrix
  (and the captured quaternion) stays the single source of truth; storing Euler per part would be a
  second drifting source (rejected per `CLAUDE.md`).
- **Read** (on select; each frame while a clip plays so the inspector tracks playback):
  `getMatrixTranslation(m)` → translate fields; `mat4.getRotation(q, m)` → q→Euler → rotation readout.
- **Write** (translate `input` only): rebuild the local matrix in place from the fields, preserving
  rotation and scale:
  ```ts
  mat4.getRotation(q, m); mat4.getScaling(s, m);
  mat4.fromRotationTranslationScale(m, q, [tx, ty, tz], s);
  ```
  (If a numeric rotation field is genuinely needed before the gizmo lands, accept and document the
  re-quantization; it is not a faithful editor.)
- **Two-writer consistency:** gizmo and inspector both write the same live matrix and neither caches,
  so they stay consistent. Guard the read/write feedback fight with a module-level `draggingEid`:
  re-read inspector fields only when the eid is **not** being gizmo-dragged.

---

## 5. Authoring — timeline + capture

### 5.1 New rxjs state (add to `state.ts`, mirroring `persistentState`)

```ts
type Mode = "view" | "edit" | "play";
export const editorMode$  = new BehaviorSubject<Mode>("view");
export const currentTime$ = new BehaviorSubject<number>(0);
export const isPlaying$   = new BehaviorSubject<boolean>(false);
export const loop$        = persistentState<boolean>("viewer.loop", true);
export const editingClip$ = new BehaviorSubject<Clip | null>(null);
```

- `currentTime$` is the single source of truth for the scrubber **and** the player's sample
  position. In `edit` mode it drives a `combineLatest` sample (genuinely cold — a human scrubbing).
  In `play` mode the pose write is called **directly in the frame loop** (§5.3), NOT through rxjs, to
  keep the per-frame pose out of the subject/`combineLatest` path.
- `editingClip$` holds the whole clip; edits emit a new clip object
  (`editingClip$.next({ ...clip, tracks })`). These are cold-path objects (a human clicking
  Capture), so object churn is fine — `CLAUDE.md` forbids it only in hot query loops.
- The set of animated parts is just `editingClip$.value.tracks.map(t => t.bone)` — no separate
  subject needed.

### 5.2 DOM (extend `index.html`)

- **Panel** (under the Animation select): `Edit clip` toggle (view↔edit), `New clip`, clip-name
  input, an `Add selected part` button, and a `#tracks` list (each row a captured bone + a ✕ to
  remove, delegated-click like the tree's `data-eid`).
- **Timeline strip** (new full-width row below the viewport; extend the `#app` grid to 2 rows):
  play/pause, loop checkbox, a `#scrub` range (`max = duration`), a `#duration` number input, a time
  label, a `Capture keyframe` button, and a `#markers` row of absolutely-positioned ticks
  (`left: (key.time / duration) * 100%`). Ticks rebuilt on each `editingClip$` emission (same
  `replaceChildren` + loop + append pattern as `fillAnimationOptions`/`appendNode`). Click a tick →
  `currentTime$.next(key.time)`; ✕/right-click deletes (`data-bone`/`data-time`).

### 5.3 Frame-loop split: view / edit / play (replaces the unconditional run line)

```ts
const mode = editorMode$.value;
const clip = editingClip$.value;
if (mode === "play" && isPlaying$.value && clip) {
  let t = currentTime$.value + delta;
  if (t >= clip.duration) t = loop$.value ? t % clip.duration : clip.duration;
  currentTime$.next(t);          // updates the scrubber UI
  player(0, t);                  // write the pose DIRECTLY — not via rxjs (L3)
} else if (mode === "none-or-builtin") {
  currentAnimations[selectedAnimation$.value]?.(delta); // legacy procedural path
}
engine.tick(delta);
```

The hot per-frame pose write is called **inline** in `play` mode, so it never routes through a
subject + `combineLatest` allocation 60×/s. A separate **cold** subscription handles `edit`-mode
scrub/edits (a human moving the cursor), which is the only place the `combineLatest` path is used:

```ts
combineLatest([editingClip$, currentTime$]).subscribe(([clip, t]) => {
  if (!clip || editorMode$.value !== "edit") return;
  if (draggingEid >= 0) return; // the gizmo's live pose wins while dragging
  player(0, t);
});
```

**Edit vs playback rule.** In `edit` mode the player samples to show the interpolated pose **unless
the gizmo is actively dragging**, in which case the gizmo's live write to `LocalTransform` wins (it
is the authoritative pose for the current cursor until captured). Releasing the gizmo leaves the
posed transform in place, ready for Capture. This cleanly separates "show what the clip says"
(sample) from "I am posing right now" (gizmo). During migration, a built-in procedural name still
runs via the legacy path when `selectedAnimation$` names it — one branch to delete once everything
is re-authored.

### 5.4 Capture / add-track / delete (cold paths, object-churn OK)

- `captureKeyframe()`: for each track in `editingClip$.value`, resolve eid via `bones`, decompose
  its live `LocalTransform` (§2.5), build a `Keyframe{ time: currentTime$.value, pos, rot }`,
  **upsert** it (replace if a key sits at ~`t`, else insert sorted), then run `canonicalizeTrack` on
  that track (§3.3) and `editingClip$.next(newClip)`. The marker subscription redraws ticks; the
  serializer subscription persists.
- `addTrack(eid)`: `eidToBone.get(eid)` → if present, push a new `Track{ bone, keys: [] }` (or seed
  one key at `t=0` = the part's current rest transform). Disabled for an eid with no bone. **Reject**
  a bone that a procedural closure owns for this entity (the one-writer rule, §3.6) — e.g. `root` on
  the swordsman.

### 5.5 rxjs wiring

All in the existing `subs` Subscription, `fromEvent` + `.next()`, identical to the current selectors
— scrub input → `currentTime$.next(+value)`; `currentTime$` → set scrub value + time label;
play/pause → `editorMode$`/`isPlaying$`; loop checkbox → `loop$`; duration change → update clip;
capture → `captureKeyframe()`; mode toggle → `editorMode$`.

---

## 6. Serialization & binding

- **Store:** one localStorage key via `persistentState` (no new mechanism):
  ```ts
  export const clipsStore$ = persistentState<Record<string, Clip>>("viewer.clips", {});
  ```
  Map key = `${entityId}/${name}`. Autosave: `editingClip$.subscribe(c => c && clipsStore$.next({
  ...clipsStore$.value, [`${c.entityId}/${c.name}`]: c }))`. `persistentState` already writes
  through to localStorage on every `.next` — no Save button.
- **Bound to entity TYPE via stable bone ids.** `Clip.entityId` = `EntityDef.id`; tracks key by bone
  name. A clip is independent of any rebuild's eids and of the viewer scale (§2.4).
- **Selector integration + name-collision rule.** `fillAnimationOptions` currently lists
  `[NONE, ...Object.keys(currentAnimations)]`. Extend it to also append authored clip names for the
  current `selectedEntityId$.value` filtered from `clipsStore$.value`. **Collision policy:** an
  authored clip whose name equals a built-in (e.g. `idle`) **overwrites** the procedural entry in the
  Record (`animations[clip.name] = makeClipPlayer(...)`). Make this explicit in the UI — authored
  names render with a marker (e.g. a `*` suffix or a separate optgroup) so the author sees the clip
  supersedes the built-in. Selecting an authored name sets `editingClip$` and routes through the
  sample/play path (§5.3); selecting a built-in runs the legacy callback.
- **Migration / coexistence.** Built-in procedural closures keep running through the same Record. As
  each keyframeable built-in is re-authored as a clip and saved, the procedural closure can be
  deleted from the builder — subject to the one-writer rule (§3.6) and the `applyPose`/`applyColor`
  split (§1.4 / §7 Phase 4). Procedural pieces clips can't represent (bob, exp-blend, alpha fade)
  stay as code permanently.
- **Export/import (optional, lean):** the store is already JSON — an Export button =
  `download(JSON.stringify(clipsStore$.value))`; import = `clipsStore$.next(parsed)`. Add only if
  shipping clips out of a browser profile becomes needed; not required for the core loop. Until then,
  authored clips live only in the browser profile (§8.7).

---

## 7. Implementation plan (phased, MVP-first runnable slices)

**Phase 0 — stable bones (foundation; no behavior change).**
`EntityInstance.bones` in `registry.ts`; `buildUnit`/`buildLightsaber`/`buildSwordsman` publish
bones (two-step `hand` migration + namespaced merge for swordsman, §2.3). Keep existing
`animations` working unchanged. *Runnable: viewer identical, bones now available.*

**Phase 1 — player + format (validate playback before authoring exists).**
New `Animation/clip.ts` (`Clip/Track/Keyframe`, `makeClipPlayer`, `sampleTrack`, `canonicalizeTrack`).
Snapshot `restScale` in `build()` right after the tree is built (§3.1). Wire a **hand-written test
clip — a fresh 2-key arm raise on `unit/armR`** (NOT the swordsman slice, which isn't a clean track,
§1.4) into `animations`, play it via the existing dropdown + a fixed cursor clock.
*Runnable: a data-driven clip plays, slerp+canonicalization verified.*

**Phase 2 — inspector + capture + timeline (the core authoring loop, no gizmo yet).**
New rxjs state (§5.1); timeline + inspector DOM (§5.2, §4.6 — editable translate, read-only rotation
readout); edit/play frame split (§5.3, pose-write inline in play mode); capture (with
canonicalization) / add-track / delete (§5.4); localStorage store + selector integration with the
collision rule (§6). Pose translation via the numeric inspector; rotation deferred to the gizmo.
*Runnable: select → pose-translate (numeric) → capture → scrub → playback → saved & reloaded.*

**Phase 3 — gizmo (ergonomics; unblocks rotation authoring).**
Input-mode split on `down$`; CPU canvas picking (§4.3) feeding `selectedEid$`; translate handles
(§4.4) reusing §6 math; single screen-facing rotate ring → quaternion (§4.5).
*Runnable: pose translate AND rotation by dragging in the viewport.*

**Phase 4 — migrate built-ins + cleanup.**
Re-author the swordsman slice/lunge as `unit/armR`+`sword/*` clips at the blended pose (root stays
procedural, §3.6); **lossy** — the stance ease-in/out is dropped until the §3.5 blender. Before
deleting unit transform code, **split `applyPose` into a transform half and `applyColor()`**
(§1.4 L5) so the death-fade survives. Keep bob/exp-blend/alpha-fade as code. Optionally
`makeBlendedPlayer` if a real multi-stance need appears (§3.5). Remove the legacy branch from the
frame split.

---

## 8. Open questions / risks

1. **Decompose/recompose fidelity under non-uniform scale.** `mat4.getRotation`/`getScaling` assume
   no shear / ~uniform scale. All current parts are uniform-scaled (builders use scale-preserving
   `setMatrixRotate*` and uniform `applyMatrixScale`), so capture and `restScale` are both clean.
   **Flag if any part ever gets non-uniform scale** — both the captured quaternion and the rest-scale
   would be wrong.
2. **Quat slerp at the loop seam.** Resolved by `canonicalizeTrack` (§3.3) plus the author placing
   keys at both `time = 0` and `time = duration`; the timeline's duration marker makes this obvious.
3. **Bone-map churn across builder edits.** Named bones are stable across rebuilds and child reorder;
   a clip breaks only if a builder *renames/removes* a bone — at which point the track resolves to
   `-1` and is skipped (explicit absence, not a silent wrong-target). Acceptable for a prototype;
   re-author on rename. One-line note in `clip.ts`.
4. **One-writer-per-bone is a contract, not arbitration.** Confirm the boundary is acceptable: clips
   own keyframeable bones, procedural closures own spring/sine bones, and the two never overlap in a
   frame (§3.6). The swordsman root specifically stays procedural; its in-place accumulation bug
   (§1.4 side note) is therefore out of scope until/unless root motion moves to a clip.
5. **Rotate gizmo is single-axis in v1.** Only the screen-facing axis rotates; the two edge-on rings
   are deferred because the screen-angle `atan2` is ill-conditioned there (§4.5). Confirm that
   re-orbiting the camera to reach other axes is acceptable ergonomics for the prototype, or schedule
   a proper ellipse-tangent ring gizmo as a follow-up.
6. **Inspector rotation is display-only.** Rotation authoring is gizmo-only (§4.6 M1); the Euler
   readout is not a faithful editor (gimbal/aliasing on `quat→Euler→quat`). Confirm this is the
   intended constraint, or accept documented re-quantization if a numeric rotation field is wanted
   before Phase 3.
7. **Clip lifetime is the browser profile.** Authored clips live only in localStorage until an
   Export path (§6) is added — note this so authored work isn't assumed to be in the repo.
8. **Selecting a non-bone part.** Parts the builder doesn't publish as bones aren't animatable
   (capture/add-track disabled). Confirm "publish what you want to animate" is the intended
   constraint, vs. auto-publishing every child (which would reintroduce the path-fragility rejected
   in §2.2).