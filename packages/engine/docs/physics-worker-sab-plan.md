# Physics-in-Worker + SharedArrayBuffer ECS plan

> Move Rapier physics into a Web Worker; keep WebGPU rendering on the main thread.
> Two bitecs worlds (one per thread) over **one** set of component columns that
> physically live in `SharedArrayBuffer`, so both worlds read/write the SAME memory.

Status: **Steps 0–3 implemented & verified** (2026-06-28). Done and tsc/node-green:
node spikes 0c/0d; **Step 1** (transform singletons killed); **Step 2** (SAB storage seam +
`sab/registry.ts` + `sab/adoptEntity.ts` + shared `NEXT_EID` + COOP/COEP); **Step 3**
(Rapier moved into a self-clocked module worker `physics.worker.ts`; physics-only worker
world bound to the received SABs; op channel `Physics/opChannel.ts` over postMessage;
`createEngine`/`RigidShapes`/`demo`/`EngineDI` rewired to drive the worker and read the
published pose bank). Browser gate **0a** confirmed by the user. **`vite build` now SUCCEEDS**
(the prior Rapier-WASM bundling error is gone — `config.vite.ts` worker `format:'es'` +
`wasm()`/`topLevelAwait()` emit a separate ES worker chunk + `rapier_wasm3d_bg.wasm`).
A node `worker_threads` integration test (`spikes/step3-opchannel.*`) proved the op-channel
+ SAB + `adoptEntity` pipeline end-to-end across real threads (16/16; Rapier stubbed).

Remaining: **browser runtime gate** for Step 3 (boxes fall driven by the worker thread;
Rapier WASM actually loads in the ES worker at runtime; spawn/despawn/clear) — user-run.
Then **Steps 4–5** (demo polish already largely folded into Step 3; sleeping-body/tearing
hardening) and the pre-training **eid-exhaustion** fix (§10).

> **Known minor artifact (Step 3, not a blocker):** `RigidBodyState.addComponent`
> zero-fills both pose banks, so a freshly-spawned shape renders at the WORLD ORIGIN for
> the single frame (~16 ms) before the worker's first `publish()`, instead of at its spawn
> pose. Fix (optional): seed the spawn pose into both banks in `RigidShapes` / let
> `addComponent` take an initial pose.

---

## 1. Goal & non-goals

### Goal (user's exact direction)

> Вытащи физику движка (`packages/engine`) в отдельный тред; данные между тредами
> должны просто существовать в `SharedArrayBuffer` у ВСЕХ components — то есть
> фактически в двух тредах будет как бы два параллельных мира ECS.

Concretely:

- Rapier (`physicalWorld.step()` + body/collider state) runs on a **module Web
  Worker**.
- WebGPU rendering (the whole `renderer3d_2` read path) stays on the **main thread**.
- There are **two bitecs worlds** — a physics world in the worker, a render world on
  main — each with its own entity registry, queries, and component bitmasks.
- The bridge is **shared memory, not message passing**: the eid-indexed component
  data columns are backed by `SharedArrayBuffer`, so eid `N` in both worlds addresses
  the same bytes. Steady-state has **zero per-frame messages**; only structural
  changes (spawn/despawn) cross the wire.

### Non-goals (explicitly out of scope for this work)

- Rendering does **not** move to a worker. No `OffscreenCanvas`. WebGPU stays main.
- No general-purpose cross-thread ECS framework. We add the minimum: one SAB-backed
  column path, one shared-buffer registry, one ordered structural-op channel, one
  pose double-buffer.
- No `ctx.table` (sparse-set) sharing — no bridge component uses it, and it
  reallocates on grow so it is not a fixed-offset SAB candidate.
- No interpolation/extrapolation in the first cut (snap to latest published pose).
- We do **not** change `delegate.defaultSize` (30000) or any component stride. The
  whole scheme depends on layout being byte-identical on both threads.

---

## 2. Architecture overview

Chosen base: **"Leanest worker/SAB physics split"** (judge's #1). **User decisions (locked
2026-06-27)** reshape it and remove several moving parts:

1. **eid = one global Atomics counter in the CONTROL SAB, NEVER recycled.** Both worlds
   draw eids from the SAME shared monotonic counter (`Atomics.add(CONTROL, NEXT_EID, 1)`),
   so eid `N` means the same entity in both worlds **by construction** — divergence is
   structurally impossible, with no replay/mirror invariant to uphold. This is the single
   most important simplification and it directly kills the show-stopper risk.
2. **Worker self-clocks** at a fixed step (true parallelism); main snaps to the latest
   published pose bank on rAF.
3. **No single-thread fallback. Worker+SAB is the ONLY path, including headless/training.**
   In node (ppo training) `SharedArrayBuffer` + `worker_threads` are available natively and
   `crossOriginIsolated` is browser-only (not required there). One code path everywhere.

Because eids are never recycled, the following parts from the original design are **dropped**:

- ~~per-eid `Generation` Int32 guard column~~ — no recycle ⇒ no recycled-eid hazard.
- ~~despawn ack-ring + generation-tagged acks~~ — main can `removeEntity` immediately;
  a late worker pose-write lands in an abandoned row that no query reads (harmless).
- ~~the "mirror every entity incl. lights, assert `addEntityId()===expected`" invariant~~ —
  unnecessary once eids come from the shared counter, not from per-world recycle history.

Retained grafts:

- A single shared, ordered **componentRegistry** consumed by both worlds so bitecs
  bitflag assignment for the **bridge** components is identical.
- A dev-mode **assertion** that bridge columns are SAB-backed on both threads
  (`col.buffer.buffer instanceof SharedArrayBuffer`).
- A per-publish **`physicsTimeMs` stamp** in the control SAB so interpolation can be
  added later without re-architecting (we do not ship interpolation now).

> ⚠️ **New tension introduced by decisions 1 + 3 — eid exhaustion (see §10).** A global
> never-recycled counter against the fixed `defaultSize = 30000` cap means total *lifetime*
> spawns are capped at 30k. Fine for the demo; **training churns spawn/despawn across
> thousands of episodes and will blow the cap.** Resolution is deferred but flagged: either a
> single **shared** SAB free-list as the one eid authority (recycling without divergence,
> because it's still one shared source), or per-episode world teardown, or a larger/growable
> capacity. Do not ship training on a monotonic counter without one of these.

### Topology

```
 ┌──────────────────────────── MAIN THREAD ────────────────────────────┐
 │  RENDER bitecs world  (eid ALLOCATION AUTHORITY)                     │
 │                                                                      │
 │  WebGPU device / canvas / context  (RenderDI)                        │
 │  rAF loop → tick(delta):                                             │
 │    seq = Atomics.load(CONTROL, SEQ)                                  │
 │    applyRigidBodyToTransform( pose read-bank → LocalTransform.matrix)│
 │    execTransformSystem (Local → Global)                              │
 │    updateLights / shapeSystem.prepare / GPU passes                   │
 │                                                                      │
 │  private (NOT shared): Shape, Color, Rope(24MB), Roundness,          │
 │     Blurness, Translucency, LightEmitter, GlobalTransform.matrix,    │
 │     camera/SunLight module globals                                   │
 └──────────────┬──────────────────────────────────┬───────────────────┘
                │ postMessage (lifecycle only)      │  SharedArrayBuffer
                │  init / spawn / despawn / ready    │  (no per-frame msgs)
                ▼                                    ▼
 ┌──────────── WORKER THREAD ─────────┐   ┌──────── SHARED MEMORY ──────────┐
 │  PHYSICS bitecs world              │   │  DATA SAB (fixed, never grows):  │
 │                                    │   │   RigidBodyState.position  *2bank│
 │  Rapier WASM heap + World          │◄─►│   RigidBodyState.rotation  *2bank│
 │  initPhysicalWorld (Z-up gravity,  │   │   RigidBodyState.linvel  (1 bank)│
 │     reserveHandleZero)             │   │   RigidBodyState.angvel  (1 bank)│
 │  pid→eid Map (worker-only)         │   │   RigidBodyRef.id                │
 │  self-clocked fixed-step loop:     │   │   Height                         │
 │    drain structural-op ring        │   │  CONTROL SAB (Int32Array):       │
 │    physicalWorld.step()            │   │   NEXT_EID  (shared eid counter) │
 │    syncRigidBodyState → back bank  │   │   SEQ, physicsTimeMs,            │
 │    Atomics.store(SEQ,++seq)+notify │   │   op-ring head/tail              │
 └────────────────────────────────────┘   └──────────────────────────────────┘
```

**Why this split**: `RigidBodyState` (pose) is the only data the worker produces that
the renderer consumes; `RigidBodyRef.id` + `Height` round out the bridge. Everything
else is render-only and stays main-private — saving ~30 MB and keeping render data off
the worker. `applyRigidBodyToTransform` stays on **main** (it reads `Height` and
produces a render component), so `LocalTransform.matrix` need **not** be shared.

---

## 3. Storage refactor (SAB-backed columns + registry)

### 3.1 The single seam: `NestedArray` / `TypedArray` over a SAB

Today (`renderer3d_2/src/utils.ts`):

- `NestedArray` ctor line 52: `this.buffer = new kind(batchLength * batchCount)`.
- `TypedArray.fXX(length)` = `new Float64Array(length)` etc. (lines 26–34).

All accessors (`get`/`set`/`getBatch`/`setBatch`/`fill`) only index `this.buffer`, so
re-pointing `this.buffer` at a view over a `SharedArrayBuffer` changes **nothing**
downstream. `getBatch` returns `this.buffer.subarray(...)`, and a subarray over an
SAB-backed typed array is itself SAB-backed (zero-copy) — confirmed safe.

Add a construction path that accepts an existing buffer + byte offset:

```ts
// NestedArray: new overload
constructor(kind, batchLength, batchCount, opts?: { sab: SharedArrayBuffer; byteOffset: number } | ArrayLike<number>)
// when opts.sab present: this.buffer = new kind(opts.sab, opts.byteOffset, batchLength * batchCount)
// else: existing behavior (new kind(len), optional seed)
```

Mirror the same for `TypedArray.fXX/u8` (a `fromSAB(sab, byteOffset, length)` variant).
Keep the existing length-only ctor intact — the single-thread fallback and all
non-bridge columns still use it.

### 3.2 The shared-buffer registry

New module: `renderer3d_2/src/sab/registry.ts`.

- A **deterministic layout table**: an ordered list of
  `{ name: "Component.field", ctor, stride, banks }` entries. `name` is a stable
  string (NOT JS-object identity), e.g. `"RigidBodyState.position"`,
  `"RigidBodyState.rotation"`, `"RigidBodyRef.id"`, `"Height.value"`. Byte offsets are
  computed by a **pure function** of `delegate.defaultSize` + stride + banks, identical
  on both threads — no negotiation. (The `Generation` column from the original draft is
  dropped — eids are never recycled, §2.)
- `banks: 2` for the double-buffered pose columns (position, rotation); `banks: 1`
  for everything else. The layout reserves `banks * stride * defaultSize * bytesPer`
  per column.
- `allocate()` (main only): asserts `globalThis.crossOriginIsolated`, builds **one
  DATA SAB** sized to the cumulative layout + **one CONTROL SAB** (small `Int32Array`).
- `bindFromSAB(dataSab, controlSab)` (both threads): returns view(s) per column name.
  The worker calls this with the **received** SABs so its views point at the identical
  bytes.

Scope (what is shared vs private):

| Column                       | Shared? | Banks | Notes                              |
|------------------------------|---------|-------|------------------------------------|
| `RigidBodyState.position`    | yes     | 2     | hot pose, double-buffered          |
| `RigidBodyState.rotation`    | yes     | 2     | hot pose quat, double-buffered     |
| `RigidBodyState.linvel`      | yes     | 1     | debug only; tearing tolerated      |
| `RigidBodyState.angvel`      | yes     | 1     | debug only; tearing tolerated      |
| `RigidBodyRef.id`            | yes     | 1     | worker backfills pid; main reads   |
| `Height.value`              | yes     | 1     | read by applyRigidBodyToTransform  |
| `LocalTransform.matrix`      | no      | —     | main-private (applyRBT runs main)  |
| `GlobalTransform.matrix`     | no      | —     | main-private (composed on main)    |
| Shape/Color/Rope/Roundness/Blurness/Translucency/LightEmitter | no | — | render-only, main-private |

### 3.3 Worker reconstructs views

`createEngineWorld` gets a variant / option. The worker side:

1. receives `{ dataSab, controlSab, layoutVersion }` via the init message,
2. asserts `layoutVersion` matches its own computed layout (drift guard),
3. binds `RigidBodyState` / `RigidBodyRef.id` / `Height` to the
   received SABs via `registry.bindFromSAB`,
4. does **not** allocate the render-only columns at all (skip flag), and
5. (dev) asserts `col.buffer.buffer instanceof SharedArrayBuffer` for each bridge col.

### 3.4 LocalTransform / GlobalTransform — kill the module singletons

This is the **#1 silent-failure hazard**. `Transform.ts:18-19` exports
`LocalTransform` / `GlobalTransform` as module-level singleton instances, and
`world.ts` spreads those same object references into every world. In a single JS realm
both worlds alias the same `NestedArray` — which **masks the bug** in any in-process
test. In a real worker the module re-executes and the singleton is re-instantiated
over private memory, so transforms silently stop sharing.

Required change (independent of the worker, testable in-process — see Step 1):

- Remove `export const LocalTransform` / `GlobalTransform` (lines 18–19). Keep only the
  factories `createLocalTransformComponent()` / `createGlobalTransformComponent()`.
- `addTransformComponents` (lines 21–35): drop the `?? LocalTransform` singleton
  fallback — always read `world.components.LocalTransform/GlobalTransform`; throw if
  absent (a world that lacks them is a bug, not a fallback case).
- `createRenderComponents` (`renderer3d_2/src/ECS/world.ts`) calls the factories
  per-world. Since `LocalTransform`/`GlobalTransform` are main-private here, the worker
  variant can skip them entirely.
- **Grep** `LocalTransform` / `GlobalTransform` across `renderer3d_2` and `engine` for
  any direct importer of the singleton and convert each to read from
  `getRenderComponents(world)` / `world.components`.

---

## 4. Structural-change protocol (two worlds, identical eids)

### 4.1 What bitecs 0.4 supports / does not

- Component VALUE columns are the only thing we control and put in SAB. bitecs's own
  state — `EntityIndex` (id allocation/recycle), `entityMasks` (presence bitmasks),
  `entityComponents`, query `SparseSet`s — is **per-world plain JS** and is **not**
  shareable. Putting data in SAB makes the bytes visible in both worlds, but neither
  world *knows* an entity exists or which components it has unless the same
  `addEntity`/`addComponent`/`removeComponent`/`removeEntity` calls are replayed there.
- **No `addEntityWithId`** in the 0.4 exports. You cannot ask a world to adopt a
  specific eid via public API.
- `createWorld` accepts an external `entityIndex`, BUT `createEntityIndex` builds
  `dense`/`sparse` as **plain JS Arrays** (not typed arrays), so the index object is
  **not** transferable/shareable across threads. The "one shared id authority object"
  escape hatch is structurally **closed**.
- `addEntityId` **recycles** from `dense[aliveCount]` before `++maxId`. So identical
  eids require identical add **and** remove/recycle **history** on both worlds, not
  just identical add order.

### 4.2 Chosen model (LOCKED): shared SAB eid counter, no recycle

The user picked the cleanest authority: **a single `NEXT_EID` Int32 in the CONTROL SAB
is the one and only eid source for both worlds.** To create an entity, whichever side
originates it does `eid = Atomics.add(CONTROL, NEXT_EID, 1)` and then `addEntityId(ctx
.entityIndex, eid)`-equivalent on its own world; the other world is told that exact eid
over the op channel and adopts the same number. Because the counter is monotonic and
shared:

- eid `N` addresses the same logical entity in both worlds **by construction**. There is
  no "identical remove/recycle history" requirement, because **nothing is ever
  recycled** — the bitecs per-world `dense[]` recycle path is bypassed for the eid value.
- A render-only entity (a light with no Rapier body) simply consumes a counter value and
  is created only in the render world; the worker never needs a no-op mirror of it,
  because the next physics entity still pulls a fresh, higher counter value. The two
  worlds' eid spaces can never collide or drift.
- **eids are globally unique forever.** Trade-off: total *lifetime* spawns are bounded by
  `defaultSize` (30000). Acceptable for the demo; **must be revisited for training** (§2
  warning, §10).

Adoption mechanism — **RESOLVED by Spike 0d (`packages/engine/spikes/eid-adoption.mjs`,
21/21 green).** ⚠️ The previously-assumed `addEntityId(ctx.entityIndex, eid)` **does not
exist** in bitecs 0.4: `addEntityId(index)` takes no eid arg (it allocates/recycles
internally) and isn't even re-exported. The proven mechanism is a small custom id-path,
shipped at `packages/renderer3d_2/src/sab/adoptEntity.ts`:

- Both worlds use `createEntityIndex()` with **`versioning:false`** so `eid === raw id`
  (required by the counter).
- `adoptEntity(world, eid)` reaches `world[$internal]` (the exported, stable `$internal`
  Symbol) → `entityIndex` and writes the caller-chosen eid directly:
  `dense[aliveCount]=eid; sparse[eid]=denseIndex; aliveCount++; maxId=max(maxId,eid)`,
  then replays `addEntity`'s per-entity init (`entityComponents.set(eid, new Set())`).
  It never touches the recycle branch. Guards: throws on `versioning:true` and on
  double-adopt of a live eid.
- Coupling (stable across 0.4, in the published `.d.ts`): the `$internal` Symbol, the
  `EntityIndex {dense,sparse,aliveCount,maxId,versioning}` shape, and `entityComponents`.
- **Documented gap**: the internal `notQueries` refresh (for `Not(...)` queries) is not
  replayed — inert today (no `Not()` queries on bridge entities); revisit if added.

The dev guard from §4.2 (assert adopted eid === op-carried counter value) still applies.

Dev-mode guard: assert the adopted eid equals the counter value the op carried (catch a
desync loudly), and assert bridge columns are SAB-backed on both threads.

### 4.3 The op channel

A small SAB ring (`Int32Array`, single-producer → single-consumer), with Atomics
head/tail in the CONTROL SAB. Records are fixed-layout `[opcode, eid, aux0, aux1]`.
Spawn carries an index into a parallel **spawn-params SAB** (type, x, y, z,
half-extents/radius, density, bodyType). Opcodes: `SPAWN_BODY`, `DESPAWN_BODY` (the
bridge component set per body is fixed, so component add/remove is implied; add explicit
`ADD_COMPONENT`/`REMOVE_COMPONENT` only if a variable set is ever needed).

**No ack ring is needed** (dropped — was only there to gate eid recycle, which no longer
happens). Despawn is fire-and-forget: main `removeEntity`s its render entity immediately;
the worker `removeRigidBody` + `removeEntity`s when it drains the op. Any worker pose
write to that eid's row in between lands in a row no query reads — harmless, never reused.

The op ring is drained by the worker **at its phase boundary, before `step()`** — never
mid-query.

> Simpler first cut allowed: batched `postMessage({type:'structural', ops:[...]})` per
> frame instead of the SAB op-ring. The SAB ring is the steady-state optimization;
> postMessage is fine while spawn/despawn volume is low (the demo).

### 4.4 Component registration order (graft)

Both worlds must register components in **identical order** or bitflag/generationId
diverge. The project's `defineComponent` also registers a **hidden shadow component
per `obs()` reactive setter** (`utils.ts` `localObs` calls `addComponent(world, eid,
setData)`), assigned bitflags lazily in registration order. The worker **skips
render-only components**, so it would never register those obs-shadows — guaranteeing
divergent bitflag assignment **if any code ever compares masks across threads**.

Mitigation: we **never compare entityMasks across threads** (presence is mirrored via
the op channel, not read from shared bytes), so divergent bitflags are harmless *as
long as that invariant holds*. To make it robust, route both worlds' bridge-component
registration through a single ordered `componentRegistry` array so the **bridge**
components at least line up; render-only/obs-shadow ordering is irrelevant because the
worker neither registers nor queries them. Document this invariant loudly.

---

## 5. Synchronization & tearing

### 5.1 The hazard

`RigidBodyState.update` writes position(3) + rotation(4) + linvel(3) + angvel(3) as 13
separate non-atomic f64 stores; the renderer reads them concurrently. f64 stores are
not individually atomic across threads, and a 7-float pose is certainly not. Result:
torn reads (new position with old/half-written quaternion → visible rotation pops).

### 5.2 Chosen model: double-buffer the pose + one Atomics seqcount

- Allocate `RigidBodyState.position` and `RigidBodyState.rotation` with **2 banks** in
  the DATA SAB.
- Worker each step writes the just-stepped pose into back bank `b = seq & 1`, then
  `Atomics.store(CONTROL, SEQ, ++seq)` + `Atomics.notify(CONTROL, SEQ)` (notify is a
  no-op for main but harmless) + writes `physicsTimeMs` into CONTROL.
- Main, once per rAF at frame start: `seq = Atomics.load(CONTROL, SEQ)`,
  `readBank = (seq - 1) & 1` (last fully-published), reads a whole consistent pose set
  from that bank in `applyRigidBodyToTransform`.
- **Atomic unit = the single `SEQ` Int32.** The f64 pose copies are non-atomic but
  made consistent by the publish/read bank flip. No per-element lock, no main-thread
  blocking. `Atomics.wait` is **forbidden on main** — main polls `Atomics.load` each
  rAF.
- `linvel`/`angvel` stay single-buffered (debug-only; tearing tolerated, documented).

### 5.3 Sleeping-body hole (must resolve — see Spike 5)

`syncRigidBodyState` skips sleeping bodies. A double-buffer publishes one seq after
writing all *awake* bodies; a sleeping body's row in the back bank is **stale** (last
written in the other bank), so after a bank flip the renderer reads a sleeping body
from a bank that never received its pose → it teleports to an old/zero pose.

Resolution (pick in Spike 5):
- **(a)** On bank flip, the worker copies the previous bank's row forward for bodies it
  skipped this step (cheap memcpy of the asleep rows), OR
- **(b)** Write ALL bodies (including sleeping) into the back bank every step (remove
  the sleep-skip in the SAB path; sleeping bodies' pose is constant so the write is
  cheap), OR
- **(c)** Keep the pose **single-buffered** and instead carry a per-entity seqlock —
  rejected (reader spin + per-entity branch).

Recommended default: **(b)** — simplest, keeps both banks always-complete, the write of
a constant pose is negligible vs. the step itself.

---

## 6. Worker boundary & message protocol

### 6.1 What runs where

**Worker** (`engine/src/physics.worker.ts`, NEW):
`initPhysicalWorld` (Rapier WASM, Z-up gravity, `reserveHandleZero`); the physics
bitecs mirror world (bridge columns bound to received SABs, render columns skipped);
the **pid→eid `Map`** (`RigidBodyRef.mapPhysicalIdToEntityId` moves worker-only);
`createBody`/`createRigid` on spawn; the self-clocked loop: drain op ring → `step()` →
`syncRigidBodyState` (→ back bank) → publish SEQ; despawn acks.

**Main** (`createEngine.ts` + render path): WebGPU device/canvas/context; render bitecs
world; eid allocation authority; op-ring producer; camera/SunLight module globals; the
render frame including `applyRigidBodyToTransform` (reads published pose bank + `Height`
→ writes main-private `LocalTransform.matrix`), `execTransformSystem`, `updateLights`,
`shapeSystem.prepare`, GPU passes.

### 6.2 Message protocol (tiny, lifecycle only)

- main → worker: `{ type:'init', dataSab, controlSab, spawnParamsSab, layoutVersion }`
  once; structural ops (via SAB op-ring **or** batched `postMessage` first-cut).
- worker → main: `{ type:'ready' }` once; `{ type:'error', ... }`; despawn-acks (via
  SAB ack-ring or postMessage).
- **No per-frame data messages.** Pose lives in SAB; readiness is the Atomics seqcount.

### 6.3 Spawn / despawn flow (demo lil-gui path)

`demo.ts` `spawnOne` → `RigidShapes.ts` `createRigidBox/Sphere`. Today these run
synchronously on main and return `[eid, pid]`; physics moves async, so:

**Spawn**:
1. Main pulls `eid = Atomics.add(CONTROL, NEXT_EID, 1)` and creates the render entity
   fully at that eid (`createRectangle`/`createSphere` set Shape/Color/Height/
   LocalTransform — all main-private or shared-`Height`).
2. Main `RigidBodyRef.addComponent(world, eid)` with `id = 0` placeholder (handle-0
   sentinel = "body not yet materialized"); `RigidBodyState.addComponent` (identity).
3. Main appends `SPAWN_BODY(eid, paramsSlot)` to the op channel.
4. Worker drains at its phase boundary: adopts the same `eid` on its world (asserts it
   matches the op), creates the Rapier body + collider, gets the pid, writes
   `RigidBodyRef.id[eid] = pid` **directly into the shared column**, registers pid→eid in
   its local Map. From the next step the pose flows through the shared bank automatically.

`createRigidBox/Sphere` now **return `eid` only** (drop the pid from the synchronous
return — graft from P3). The render entity is fully valid before the worker replies and
simply sits at its spawn transform until the first pose publish, so there is no
dangling-entity risk. Callers that need the pid read `RigidBodyRef.id[eid]` (0 =
pending) — but the demo only used pid for despawn lookup, which now goes through eid.

**Despawn** (`demo.ts` `clearDynamic`) — fire-and-forget, no ack:
1. Main appends `DESPAWN_BODY(eid)`, then immediately `RigidBodyRef.clear(eid)` +
   `removeEntity(render world)`. The eid is **never reused**, so there is nothing to gate.
2. Worker drains at its phase boundary: `removeRigidBody`, `Map.delete(pid)`,
   `removeEntity` on its world.

Between (1) and (2) the worker may publish one more pose into the despawned eid's row;
no query reads it (the entity is gone from both worlds) and the eid is never recycled, so
it is harmless. This holds even if the worker self-clocks faster than rAF and produces
several steps + new spawns within one render frame — new spawns take fresh higher counter
values that can never collide with the dead eid.

Audit: grep every main-thread use of `engine.physicalWorld` (`demo.ts`, `setupScene`,
`buildGui`, `createGround`, `clearDynamic`, any debug). On main, `physicalWorld` no
longer exists — `EngineDI.physicalWorld` becomes `undefined` (or a thin proxy) and all
its uses route through the spawn/despawn channel.

### 6.4 Cross-origin isolation (COOP/COEP)

`SharedArrayBuffer` requires `crossOriginIsolated === true`, which needs response
headers. `config.vite.ts` currently sets **no** headers. Add to both `server` and
`preview`:

```ts
// config.vite.ts
export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  optimizeDeps: { exclude: ["@dimforge/rapier3d-simd"] },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  // worker bundle must be ES so vite-plugin-wasm + top-level-await apply inside it:
  worker: { format: "es", plugins: () => [wasm(), topLevelAwait()] },
});
```

- `index.html` needs **no** change (COEP is server-sent, not a meta tag).
- `COEP: require-corp` makes the page reject any cross-origin subresource lacking
  CORP/CORS. Rapier WASM is bundled same-origin (fine). **Spike 0** must confirm the
  existing WebGPU/voxel/SDF render path still works under these headers before any
  architecture work.
- The production host that serves the built engine must send the same headers, or SAB
  fails only in production. With **no fallback** (user decision), a missing-isolation
  startup must **fail loud** with a clear message (`crossOriginIsolated === false → set
  COOP/COEP`), not silently degrade.
- **Headless/training (node) needs no COOP/COEP** — `SharedArrayBuffer` + `worker_threads`
  are always available in node; `crossOriginIsolated` is a browser concept. The worker+SAB
  path is the same code; only the worker construction differs (`new Worker(url,{type:
  'module'})` in the browser vs `node:worker_threads` in training). Abstract that one seam.

---

## 7. Frame cadence

- **Worker self-clocks** physics at a fixed timestep (Rapier's internal fixed step;
  note `physicalWorld.step()` already ignores any passed delta today, so this is a
  formalization, not a behavior change). The worker runs its own accumulator loop
  (`setTimeout`/`MessageChannel` ping, or `Atomics.wait`-with-timeout for tighter
  pacing) — `requestAnimationFrame` is unavailable in workers. Cap substeps per wake to
  avoid spiral-of-death.
- **Main renders on rAF**; at frame start it reads the latest published pose bank
  (Section 5.2) and snaps to it. The two clocks are decoupled (true parallelism) and
  rendezvous only through `SEQ`. Neither blocks the other: the worker free-runs if rAF
  is slow; the renderer re-displays the last published bank if physics stalls.
- `createEngine.tick(delta)` drops `physicalFrame()` entirely; `applyRigidBodyToTransform`
  (now reading the published bank) moves into the render frame ahead of
  `execTransformSystem`.
- **Interpolation: deferred.** We write `physicsTimeMs` per publish so lerp(position) +
  slerp(rotation) between the two banks can be added later as a local change if judder
  appears at mismatched rates. Snap-to-latest ships first (tear-free, simplest).

---

## 8. Staged implementation plan

Each step is independently verifiable. Steps 0–1 are pure prerequisites and carry no
worker risk; the worker only appears at Step 3.

### Step 0 — Spikes (gate everything). No production code.
- **0a (headers gate)**: add COOP/COEP to `config.vite.ts`, run the **existing**
  engine demo unchanged. Verify `globalThis.crossOriginIsolated === true`, WebGPU still
  inits, the voxel/SDF scene renders identically, Rapier WASM still loads. *If
  require-corp breaks the render path, the whole approach is blocked.*
- **0b (Rapier-in-worker under Vite)**: bare module worker
  (`new Worker(new URL('./physics.worker.ts', import.meta.url), {type:'module'})`),
  import `@dimforge/rapier3d-simd` inside it, run `initPhysicalWorld` + a few `step()`s,
  postMessage a position back. Confirm `vite-plugin-wasm` + `top-level-await` apply to
  the **worker** bundle.
- **0c (SAB NestedArray)**: add the `(sab, byteOffset)` ctor path; allocate one column
  on a SAB on main, postMessage the SAB to a worker, write a quaternion in the worker,
  read it on main; assert byte-identical and `col.buffer.buffer instanceof
  SharedArrayBuffer` on **both** threads.
- **0d (shared-counter eid adoption)**: prove a world can be forced to adopt a specific
  eid pulled from a shared `Atomics` counter — via `addEntityId(ctx.entityIndex, eid)`
  (internal) or a tiny custom id path with `versioning:false`. Drive two worlds where one
  creates render-only entities the other never sees; confirm the shared counter keeps eid
  meaning identical with **no recycle** and no divergence. Confirm `removeEntity` does NOT
  return the eid to a reuse pool that the counter would later re-hand-out. Pick the
  adoption mechanism here.
- **Verify**: each spike runs green before proceeding. 0a and 0d are hard gates.

### Step 1 — Kill transform singletons, single-threaded (no SAB yet).
- **Files**: `renderer3d_2/src/ECS/Components/Transform.ts` (remove singletons +
  fallback), `renderer3d_2/src/ECS/world.ts` (per-world factory calls), every grep hit
  of `LocalTransform`/`GlobalTransform` direct import.
- **Verify**: existing single-thread demo renders identically. Flushes out all direct
  singleton importers.

### Step 2 — SAB-back the bridge storage, single-threaded.
- **Files**: `renderer3d_2/src/utils.ts` (SAB ctor path), `renderer3d_2/src/sab/
  registry.ts` (NEW), `renderer3d_2/src/sab/componentRegistry.ts` (NEW),
  `engine/src/ECS/components.ts` + `engine/src/ECS/createEngineWorld.ts` (bind bridge
  columns via registry; `skipRenderOnly` flag), `engine/src/ECS/Components/
  RigidBodyState.ts` (pose getters/setters address active bank), `RigidBodyRef.ts`
  (document worker-only Map). Add a `crossOriginIsolated` **fail-loud guard** at startup
  (no fallback — user decision); add the shared `NEXT_EID` counter in the CONTROL SAB and
  route all eid allocation through it.
- **Verify**: demo still runs with physics inline on main (worker not split out yet), but
  bridge columns now live on a SAB (assert SAB-backed in dev) and eids come from the
  shared counter. No behavior change yet — this is the storage + eid-source swap.

### Step 3 — Move Rapier to the worker; wire structural-op channel + pose double-buffer.
- **Files**: `engine/src/physics.worker.ts` (NEW), `engine/src/sab/opChannel.ts`
  (NEW: op-ring + ack-ring, or postMessage first-cut), `engine/src/createEngine.ts`
  (spawn worker, allocate+post SABs, drop `physicalFrame`, move
  `applyRigidBodyToTransform` into render frame reading the published bank),
  `createRigidBodyStateSystem.ts` (runs in worker, writes back bank, resolve sleeping
  bodies per Spike 5), `createApplyRigidBodyToTransformSystem.ts` (reads read bank).
- **Verify**: a box spawned on main falls under gravity in the worker and the render
  entity moves on screen; `RigidBodyRef.id[eid]` is backfilled by the worker; dev
  assertion "adopted eid === counter value from the op" holds.

### Step 4 — Demo spawn/despawn over the async protocol.
- **Files**: `engine/src/ECS/Entities/RigidShapes.ts` (return eid only, post spawn,
  drop PhysicalWorld arg), `engine/src/demo.ts` (`Spawned = {eid}`, `clearDynamic`
  posts despawn + ack-gated recycle), `engine/src/DI/EngineDI.ts`
  (`physicalWorld` no longer main-thread).
- **Verify**: lil-gui spawn/despawn/clear works; rapid spawn→despawn→respawn does not
  corrupt (no recycle ⇒ fresh eids, dead rows never re-read); count readout correct.

### Step 5 — Sync/tearing hardening.
- Resolve the **sleeping-body** publish hole (Spike 5 decision (b) recommended).
- Stress the double-buffer under a worker stepping faster than rAF; confirm no torn
  quaternions, no teleporting sleeping bodies.
- (The despawn/recycle race is **eliminated by design** — eids are never recycled — so no
  separate spike is needed; just confirm a late pose-write into a despawned row is inert.)
- **Verify**: hostile-schedule prototype green; visual smoke test shows no pops.

---

## 9. Risks & required spikes (from adversarial review)

| Risk | Mitigation |
|------|------------|
| ~~**eid recycle divergence**~~ — **ELIMINATED by the shared-counter decision.** eids come from one `Atomics` counter and are never recycled, so the two worlds cannot drift. | Shared `NEXT_EID` in CONTROL SAB; force-adopt the value on each world (Section 4.2). **Spike 0d** picks the adoption mechanism (`addEntityId(ctx.entityIndex,eid)` vs custom path). |
| **External EntityIndex not shareable** — `dense`/`sparse` are plain JS Arrays. | Moot with the shared counter — we don't share the index object, only the counter. |
| **Transform module singletons mask the bug in-process** then break in a real worker. | Step 1 removes singletons + fallback before any sharing; grep all importers. |
| **obs() shadow components break component-order parity** when worker skips render-only. | Never compare entityMasks across threads; presence is mirrored via the op channel, not read from shared bytes. Route bridge registration through a single ordered `componentRegistry`. Document the invariant. |
| **Torn 7-float pose reads.** | Double-buffer pose + single Atomics seqcount (Section 5.2). |
| **Sleeping-body publish hole** — skipped bodies leave stale rows after bank flip. | **Spike 5**; recommended: write all bodies (incl. sleeping) into the back bank every step. |
| **eid exhaustion** — never-recycled counter vs fixed 30k cap; training churn overruns it. | Demo is safe. **Before training**: a single *shared* SAB free-list as the one eid authority (recycle without divergence), per-episode world teardown, or larger/growable capacity. Flagged in §2 / §10. |
| **Headless/training path** — must run with no renderer/WebGPU. | **No fallback** (user decision): worker+SAB everywhere. node provides SAB + `worker_threads` natively (no COOP/COEP); abstract only the worker-construction seam. Fail loud if isolation missing in the browser. |
| **COOP/COEP side effects** on WebGPU/assets. | **Spike 0a** smoke-tests the existing render path under headers before any work. |
| **Memory doubling** if worker private-allocs a bridge column. | Registry binding is the #1 correctness point; dev assert `col.buffer.buffer instanceof SharedArrayBuffer` on both threads after init. |
| **Vite worker + WASM bundling** (TLA unsupported in classic workers). | `worker.format: 'es'` + plugins mirrored; **Spike 0b**. |

**Required spikes** (Step 0): 0a headers gate, 0b Rapier-in-worker, 0c SAB NestedArray,
0d shared-counter eid adoption. Plus Spike 5 (tearing + sleeping body) before Step 5
sign-off. (The despawn-race spike is dropped — no recycle, no race.)

---

## 10. Decisions

### Locked (user, 2026-06-27)

- **eid model** → **one shared `Atomics` `NEXT_EID` counter in the CONTROL SAB, never
  recycled.** Kills the divergence show-stopper structurally (§4.2).
- **Worker pacing** → **self-clocked fixed step** (true parallelism); main snaps to the
  latest published bank on rAF (§7).
- **Headless/training topology** → **no single-thread fallback; worker+SAB is the only
  path everywhere.** node provides SAB + `worker_threads` natively; abstract just the
  worker-construction seam. Fail loud if browser isolation is missing (§6.4).

### Still open / recommended defaults

- **Structural channel: SAB op-ring vs. batched postMessage** — start with **batched
  postMessage** at phase boundaries; move to the SAB ring only if spawn/despawn volume
  makes messaging a bottleneck. *Default: postMessage first.*
- **Sleeping-body publish** — **write all bodies (incl. sleeping) into the back bank
  every step** (Spike 5 confirms). *Default: (b).*
- **Interpolation** — **defer** (snap to latest bank); keep the `physicsTimeMs` stamp so
  lerp/slerp is a later local change. *Default: defer.*
- **Pose double-buffer scope** — double-buffer **position + rotation only**; keep
  `linvel`/`angvel` single-buffered (debug-only). Confirm no render/effects code does
  matrix math on velocities. *Default: pose only.*
- ~~**eid adoption mechanism**~~ — **RESOLVED** (Spike 0d): custom `adoptEntity()` over
  `world[$internal]` with `versioning:false`; the assumed `addEntityId(ctx.entityIndex,
  eid)` does not exist. Shipped at `renderer3d_2/src/sab/adoptEntity.ts` (§4.2).

### ⚠️ Required follow-up before training (consequence of the locked eid + headless choices)

- **eid exhaustion.** A never-recycled counter against `defaultSize = 30000` caps *total
  lifetime* spawns at 30k. The demo is safe; **training (worker+SAB everywhere, high
  spawn/despawn churn across thousands of episodes) will overrun it.** Resolve before
  wiring ppo_engine: (a) a single **shared** SAB free-list as the one eid authority —
  recycling without divergence because it stays one shared source; (b) tear down + rebuild
  the worlds per episode; or (c) raise/grow capacity. *Recommendation: (a) when training
  lands; keep the monotonic counter for the demo now.*
```
