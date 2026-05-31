# Multi-world migration plan — Physics / Render / Slot / Fx / Game / Sound + Bridge

> Goal: split the single bitecs world into several worlds grouped **by hot-loop
> archetype**, so the cache-critical iterations (physics sync, render draw) walk
> arrays that are ~100% populated. Driven by the priorities:
> **render iteration = max performance, physics iteration = max performance,
> game/business-logic iteration = doesn't matter (simplicity > perf).**
> Cross-world **identity** (the id link tables + translators) is owned by one tiny registry —
> `BridgeDI` (§6) — and **only** that; spawn/teardown, data sync, and joints live elsewhere and
> merely call Bridge to register or translate ids.

> **PRIME INVARIANT — the split is behavior-preserving. This overrides every other section.**
> Moving a component to another world changes only **where** it is stored, never **what** the
> game does. The migration MUST NOT change, for any entity: its existence and identity, the
> **rigid-body & joint topology in Rapier** (every body and every joint that exists today
> exists **1:1** afterwards — same bodies, same parent↔child pairing, same anchors, same
> collision groups), the slot graph, system logic, numeric tuning (impulses, durations,
> colors), or any observable gameplay/visual.
>
> Corollaries — read these before designing any spawn/teardown:
> - **An "atom" = exactly ONE legacy Rapier body**, re-expressed as `{PhysicsWorld physics-eid
>   + RenderWorld mirror-eid}`. An atom is **never** a merge of two legacy bodies, and a legacy
>   body is **never** demoted to "just a slot/anchor on another body."
> - A vehicle today is a **graph of jointed bodies** — hull, turret, **each** track, **each**
>   wheel, gun, … — not "hull + turret". Every one of those bodies stays its own atom; every
>   joint between them is recreated between the corresponding atoms. Control/physics systems
>   that drive a track/wheel/gun body must still find that body as an atom — **collapsing
>   sub-bodies into hull/turret anchors is exactly what silently breaks movement** (the control
>   system applies impulse to a track body that no longer exists as an independent atom).
> - A body that also **carries slots** keeps carrying them: the slot's `CarrierRef` points to
>   **that exact atom** (the track atom, the turret atom, …), not to a different "nearest" body.
> - If a design step would change body count, joint structure, anchors, or behavior, **the step
>   is wrong — fix the design, not the game.** The only acceptable verification is: the running
>   game behaves identically to pre-migration.

## 0. Why this layout (the density rule)

Components are stored as `TypedArray(delegate.defaultSize)` indexed by **raw `eid`**
(see `renderer/src/ECS/utils.ts`, `Components/Physical.ts`). Therefore:

```
hot-loop density for component X = (#eids that have X) / (max eid in that world)
```

A hole in `Shape[]` is **not** caused by an entity having many components — it is
caused by *foreign eids that lack `Shape`* sitting in the same id-space. So: **keep in
one world only entities that share the hot-loop signature.**

### The key relationship: physics ⊆ render

Every physical object is rendered, but **not** every rendered object is physical —
there are (and will be a growing number of) **bodyless visual children** (decals,
attachments, overlays), plus render-only fx. A single world can't be dense for both hot
loops at once, so **physics and render are split into two worlds linked by a mirror**.
Because physics ⊆ render, the link is an **injective map `physicsEid → renderEid`**.

### Slots are logical and cold — they get their own world

A `slot` is purely logical (not rendered, no collider): a **permanent** structural
mount point on a vehicle, holding a local anchor, size, `partType`, and an occupant
reference. A `part` is a hot physical+render atom. By count `#slots ≳ #filled_parts`, so
mixing them would put ~50% holes in the part hot-arrays. Slots are also cold (touched
only by spawn / heal / upgrade / shield management). → slots live in their **own
`SlotWorld`** (first-class, isolated; perf-neutral because cold).

## 1. Target topology (6 bitecs worlds + Rapier)

| World | Holds | Hot? | Headless |
|-------|-------|------|----------|
| **Rapier** (WASM, `GameDI.physicalWorld`) | rigid bodies, joints, contacts, integration | — | yes (unchanged) |
| **PhysicsWorld** 🔥 | physical atoms only: hulls, turrets, wheels, tracks, vehicle parts, **bullets**. **No render components.** Physics bridge + per-atom gameplay needed for contacts. Source of truth for position. | physics-sync | **yes** — render-free sim core |
| **RenderWorld** 🔥 | everything drawn: **mirrors** of physics atoms + **bodyless visual children** + the `Parent/Children` hierarchy. | draw | no |
| **SlotWorld** | logical mount points (cold): `Slot` (anchor, size, partType), occupant link. Permanent per vehicle. | no (cold) | yes |
| **FxWorld** | render-only ephemera: `TreadMark`, `VFX`, `Explosion`, `MuzzleFlash`, `HitFlash`, `ExhaustSmoke`. High churn isolated. | draw | no |
| **GameWorld** | business logic / "brain", few entities, simplicity > perf: `Vehicle`, `VehicleController`, `TurretController`, `Firearms`, `Tank`, `HeuristicsData`, `LastHitters`, **canonical** `TeamRef`/`PlayerRef`. | no (cold) | yes |
| **SoundWorld** | `Sound`, `SoundParentRelative`, `DestroyOnSoundFinish`. | no | no |

## 2. Component assignment

**PhysicsWorld** (kept 100% populated; **render-free**):
- Physics bridge: `RigidBodyRef`, `RigidBodyState`, `Impulse`, `TorqueImpulse`, `ImpulseAtPoint`, `Joint`, `JointMotor`.
- Per-atom gameplay on the contact path: `Hitable`, `Damagable`, `VehiclePart`, `VehiclePartCaterpillar`, `VehicleTurret`, `Wheel`, `WheelDrive`, `WheelSteerable`, `Track`, `Bullet`, `Obstacle` (static physical colliders are physics atoms too).
- **Cheap copy** `teamId`/`playerId` on the atom for the contact-filter fast path (canonical owner is in GameWorld — see §5).
- Lifecycle: `Destroy`, `DestroyBySpeed`, `DestroyByTimeout`, `Progress` (bullets).
- Link components (written only by Bridge, §6): `RenderRef` (→ mirror), `OwnerRef` (→ brain), `HomeSlotRef` (→ slot).

**RenderWorld** (kept 100% populated; **no physics components**):
- `Shape`, `Color`, `Roundness`, `Thinness`, `LocalTransform`, `GlobalTransform`, `Rope`.
- `Parent`/`Children` — visual hierarchy (mirror → bodyless children).
- Link component: `PhysicsRef` (→ source atom) on mirrors.

**SlotWorld**: `Slot` (anchorX/Y, width, height, partType), `OccupantRef` (→ part atom, 0 = empty).

**FxWorld**: `VFX`, `TreadMark`, `Explosion`, `MuzzleFlash`, `HitFlash`, `ExhaustSmoke`
(+ their `Shape`/`Color`/`Transform`/`Progress`).

**GameWorld**: `Vehicle`, `VehicleController`, `TurretController`, `Firearms`, `Tank`,
`HeuristicsData`, `LastHitters`, canonical `TeamRef`/`PlayerRef`, vehicle→slots/atoms id lists.

**SoundWorld**: `Sound`, `SoundParentRelative`, `DestroyOnSoundFinish`.

## 3. Systems are not owned by worlds

A "system" is just a closure over one or more worlds returning a per-frame function;
the tick owns systems, not the world. A system may freely span worlds. Rules:

- `query(world, [...])` is **per-world**; iterate your **hot/primary** world, reach the
  rest by id-link. Disjoint id-spaces ⇒ an eid from world A is meaningless as an index
  into world B — translate at every cross-world hop.
- Cross-world access = random indirection. Matters only for **hot** loops; **cold**
  systems (few entities) span worlds for free.

Example: `createVisualizationTracksSystem` iterates `Track` (PhysicsWorld, ~2/tank → cold)
and reaches `Slot.anchorX/Y` (SlotWorld) via id-link. No relocation, just translation.
(Future: that per-frame anchor animation is physics state — could move onto the
filler-part atom in PhysicsWorld to drop the cross-world hop. Not now.)

## 4. Slot ↔ part model & lifecycle

- **Carrier** (any PhysicsWorld atom): the body a slot is mounted on. Carriers are **general** —
  hull, turret, a **track**, a **wheel**, or any other legacy body that owned slots stays the
  carrier of exactly those slots. `CarrierRef` (on the slot) points to **that exact atom**. Do
  **not** re-parent a slot onto a "nearest" larger body — that changes the body graph (§ Prime
  Invariant). Carrier atoms are themselves jointed into the vehicle's body graph exactly as in
  the legacy model.
- **Slot** (SlotWorld): permanent, logical, lives for the vehicle's lifetime. Source of
  truth for **where** a part belongs (local anchor, relative to its carrier). `OccupantRef` →
  current part (0 = empty).
- **Part** (PhysicsWorld atom + RenderWorld mirror): standalone entity; remembers its home
  via `HomeSlotRef`. Two states:
  - **attached**: joint to carrier at the slot anchor; `slot.occupant = part`, `part.homeSlot = slot`.
  - **detached** (torn off): joint removed; `slot.occupant = 0`; `part.homeSlot` **kept**;
    the part becomes a free rigid body lying on the ground with full physics.
- **heal / upgrade**: take a detached part → read `HomeSlotRef` → recreate the joint at the
  slot anchor → `slot.occupant = part`. (Slot says *where*, part remembers *which slot*.)

The joint/entity work of attach/detach/restore lives in the slot/part code; it **registers the
resulting id links via `BridgeDI`** (`link`/`unlink`, §6) so the link graph stays consistent. Bridge
itself performs none of the joint/entity work.

## 5. Team / Player ownership

- **Canonical** `TeamRef`/`PlayerRef` live on the vehicle **brain** (GameWorld) — single
  source of truth, mutable.
- A **cheap read-only copy** `teamId`/`playerId` sits on each physics atom for the hot
  contact-filter path (no cross-world hop on every contact). Set once at spawn by the spawn
  factory (which then registers the atom→brain `OwnerRef` link via Bridge).
  Part team/player is static for the part's life, so the copy never needs live updates;
  a whole-vehicle capture re-stamps via the atom list.

## 6. `BridgeDI` — the cross-world id registry (and ONLY that)

A module-level singleton (like `GameDI`). Its **single** responsibility is **id synchronization
between worlds**: it owns the cross-world link tables and answers id-translation queries. That is
all it does. It does **NOT** create or destroy entities, call Rapier, copy component data, manage
joints, schedule structural changes, or run any game logic. Every such operation lives in an
ordinary factory/system that merely **calls Bridge** to register/clear a link or to translate an
id across worlds. Keeping Bridge this narrow IS the design — one tiny, auditable id map, never a
god-object.

> Storage decision: link data is stored **as ECS components** on the entities
> (`RenderRef`, `PhysicsRef`, `OwnerRef`, `HomeSlotRef`, `OccupantRef`, `CarrierRef`) for locality
> — but **only Bridge writes them**. Authority over the *id graph* = Bridge; authority over
> *everything else* (spawn, teardown, sync, joints) = the factories/systems that call Bridge.

### 6.1 What Bridge owns — the link graph
The cross-world links, stored as ECS link components (for locality) + the rapier-handle map.
**Only Bridge writes them; everyone else reads them via translators:**
- `physics ↔ render` (mirror; injective; **invariant physics ⊆ render**) — `RenderRef`/`PhysicsRef`
- `physics → brain` — `OwnerRef`
- `part ↔ homeSlot` / `slot → occupant` — `HomeSlotRef` / `OccupantRef`
- `slot → carrier` — `CarrierRef`
- `rapier handle ↔ physics` — a plain map (replaces `mapPhysicalIdToEntityId`, `Components/Physical.ts`)

### 6.2 What Bridge exposes — only two kinds of method (+ a debug check)
- **mutators** — register/clear a link: `link(kind, a, b)` / `unlink(kind, a)`. Called BY the spawn
  factories / teardown / attach-detach code right after they create or remove the underlying
  entities, bodies, and joints. Bridge only records the id pairing.
- **translators** — pure lookups, no side effects: `getRenderOf` / `getPhysicsOf` / `getBrainOf` /
  `getHomeSlotOf` / `getOccupantOf` / `getCarrierOf` / `getPhysicsByRapier`. Any cross-world system
  uses these to hop id-spaces.
- **`validate()`** (debug-only, every N frames in dev) — audits Bridge's OWN tables: physics ⊆
  render, bidirectional consistency (`render[physics[r]] === r`), no dangling targets. Fails loud.
  This checks the id graph only; it runs no game logic.

### 6.3 What is NOT in Bridge — and where each piece lives instead
These were over-scoped into Bridge before; each only **uses** Bridge's mutators/translators:
- **Spawn factories** (`spawnVehicle` / `spawnPart` / `spawnBullet` / `spawnFx`) — the re-expressed
  `createVehicleBase` / `createTrack` / `createWheel` / `createVehicleTurret` / `fillSlot` /
  `spawnBullet`. They allocate the entities in each world and create the Rapier bodies/joints,
  reconstructing the **exact legacy body+joint+slot graph 1:1** (§ Prime Invariant: hull, turret,
  **every** track/wheel, gun, … each its own atom+mirror; every joint; every slot on its **original
  carrier atom**; same anchors, motors, collision groups, per-set colors; **never** a hull+turret
  collapse) — then call `Bridge.link(...)` to register the resulting id links. Bridge creates nothing.
- **Teardown** (`destroyVehicle` / `destroyPart`) — reads links via Bridge, removes the entities,
  Rapier bodies and joints in each world, then calls `Bridge.unlink(...)`. The brain holds the
  id-lists of its owned atoms/joints/slots so the identical set is reaped (no leaks, no dangling).
- **Per-frame transform copy** — a RENDER-tick system (`mirror-sync`): for each AWAKE physics atom,
  translate physicsEid→renderEid via `Bridge.getRenderOf` and copy `RigidBodyState`(x,y,rot) →
  mirror `LocalTransform`. "Sync as needed" (skip sleeping bodies). Bridge only supplies the
  translation; the copy is the system's logic.
- **attach / detach / heal** (§4) — slot/part joint+link operations: create/remove the Rapier joint
  and set the relevant state, calling `Bridge.link/unlink` for the part↔home / slot→occupant links.
- **Deferred structural command buffer + flush** — owned by the tick (a small command queue), NOT
  Bridge. Systems enqueue spawn/teardown commands; the tick drains them at one defined point (end of
  SIM) so no multi-world structural mutation happens mid-query. Bridge holds no queue.

## 7. Tick order after the split

```
SIM tick (headless-capable; PhysicsWorld + GameWorld + SlotWorld + Rapier — NO render):
  control intent: TrackControl / WheelControl / TurretRotation
      read brain (GameWorld) + atom RigidBodyState (PhysicsWorld) -> write atom Impulse/JointMotor
  applyImpulses / applyJointMotors            -> push to Rapier             (PhysicsWorld)
  rapier.step(eventQueue)
  syncRigidBodyState                          <- Rapier -> RigidBodyState   (PhysicsWorld, DENSE)
  contact events -> Hitable damage            (pid -> atom -> OwnerRef -> brain)
  gameplay: tank-alive / shields / progress / destroy scheduling
  spawn/detach/heal/destroy -> ENQUEUE on the command buffer
  command-buffer flush                        <- atomic structural mutation; factories create
                                                 entities/bodies/joints, then call Bridge.link/unlink

RENDER tick (skipped in headless):
  mirror-sync system                          <- copy RigidBodyState -> mirror LocalTransform
                                                 (awake atoms; physicsEid->renderEid via Bridge translator)
  drain Fx spawn/despawn (command buffer)     (FxWorld)
  TransformSystem (RenderWorld): mirror . localOffset -> bodyless children   (intra-world)
  fx-update: TreadMark fade, VFX progress     (FxWorld)
  draw: fauna -> grid -> shapes(RenderWorld) -> shapes(FxWorld) -> vfx -> sandstorm
  camera (reads target atom RigidBodyState) ; sound (SoundWorld)
```

## 8. Migration steps (each step lands behavior-neutral & runnable; within a step the tree is red until complete)

Worlds are introduced **one group at a time**, not all at once. The invariant holds at every step
boundary: the game behaves identically (§ Prime Invariant). A component not yet given its own world
**temporarily co-locates** on the atom/mirror it rides today, and moves to its final world in a later
step — a behavior-neutral relocation.

**Step 0 — prep: thread `world` explicitly (mechanical, no behavior change).**
Remove `{ world } = GameDI` defaults; pass world in everywhere it's created/read:
`Components/RigidGroup.ts`, `Components/RigidRender.ts`, `Physical/createRigid.ts`,
`Entities/Vehicle/VehicleParts.ts`, renderer `Shapes.ts` / `addTransformComponents`,
sound spawners. Biggest diff; unblocks everything.

**Step 1 — `BridgeDI` + `PhysicsWorld` + `RenderWorld` (the mirror seam — the core hot-path split).**
Stand up exactly two ECS worlds + the id registry:
- `createPhysicsWorld` (physical atoms) and `createRenderWorld` (mirrors), each its own component
  subset (§2); `GameDI` gets both handles + `physicalWorld`.
- `BridgeDI` as the id registry only (§6): `RenderRef`/`PhysicsRef` link components + the
  rapier-handle map (replacing `mapPhysicalIdToEntityId`); `link`/`unlink` + translators +
  `validate()`. **Nothing else in Bridge.**
- The physics↔render split of `createRigidRectangle/Circle` (`spawnPart` factory): the atom in
  PhysicsWorld, its mirror in RenderWorld, links registered via Bridge; teardown likewise. The
  command buffer (tick-owned) so structural changes don't mutate mid-iteration.
- The `mirror-sync` system (awake-only `RigidBodyState`→mirror `LocalTransform` via Bridge
  translators, replacing `applyRigidBodyToTransform`).
- Re-point the pure systems: `syncRigidBodyState`/`applyImpulse`/`jointMotor` → PhysicsWorld;
  `TransformSystem`+`drawShape`/`drawRope` → RenderWorld.
- **Transitional co-location:** everything that is NOT a render component stays where it can ride —
  brain/gameplay components (`Vehicle`/`Controller`/`Firearms`/`Tank`/`Heuristics`/`LastHitters`/
  canonical `Team`/`Player`, plus `Hitable`/`Damagable`/etc.) sit on the **physics atom**; the
  still-non-physical entities (slots, fx, sound, bodyless children) sit in **RenderWorld**. Control/
  gameplay systems read them from there for now. § Prime Invariant: the full body+joint graph is
  preserved 1:1 — every track/wheel/gun is its own atom+mirror; no hull+turret collapse.
After Step 1 the game runs on two worlds joined by the mirror.

**Step 2 — add `SlotWorld`.**
`createSlotWorld`; carve slots out of their temporary RenderWorld home into SlotWorld with
`CarrierRef` (→ the **original carrier atom**), `OccupantRef`, `HomeSlotRef`. `createSlotEntities`
builds SlotWorld slots; `fillSlot` becomes `spawnPart` + attach (joint + `Bridge.link` for
`OccupantRef`/`HomeSlotRef`); add detach/heal (§4). Re-point slot-reading systems (e.g.
`visualizationTracks`, shield management) to SlotWorld via Bridge translators (cold → fine).

**Step 3 — the rest: `BrainWorld` + `FxWorld` + `SoundWorld`.**
- `createBrainWorld` (the **GameWorld**/brain of §1–§2): carve the brain/gameplay components off the
  physics atoms into it; set `OwnerRef` (atom→brain) + the cheap `ContactFilter` team/player copy (§5); the
  brain holds the id-lists of its owned atoms/joints/slots for teardown. `spawnVehicle` factory
  becomes its final form (brain + full atom/joint/slot graph). Re-point control/contact/camera/
  action systems to read brain via Bridge translators.
- `createFxWorld`: move fx entities + spawners out of RenderWorld via the `spawnFx` factory
  (world-space snapshot of the source); draw fx as a second SDF pass after the render-mirror pass.
- `createSoundWorld`: move sounds out; sound position resolves via Bridge (sound→brain→hull atom).
After Step 3 the full six-world topology (§1) is in place and the game runs identically.

**Step 4 — headless flag (optimization; optional, last).**
Not part of the world split — a perf/runtime lever enabled by it. Gate the RENDER tick +
RenderWorld/FxWorld/SoundWorld creation behind a `headless` flag in `createGame`; headless runs
Rapier + PhysicsWorld + BrainWorld + SlotWorld only (the render-free sim core, for future
headless/ML use). Behavior of the rendered game is unchanged.

## 9. Optional later levers (only if profiling demands)

- **Separate ProjectileWorld** — isolate bullet create/destroy churn from the atom eid range.
- **Packed component storage (sparse-set)** — replace `array[eid]` in the `defineComponent`
  factory (`renderer/src/ECS/utils.ts`) with `dense[i]` + `sparse[eid]→i` + swap-remove;
  sequential iteration dense regardless of eid holes; cost = one indirection on random access
  by eid. Risky, repo-wide; keep last.
- **Archetype/chunk store (DOTS/Flecs-style)** — the true ceiling; bitecs has no native
  support. Out of scope.

## 10. Resolved decisions & remaining risks

Resolved:
- **Behavior preservation is the prime invariant** (see top): the split relocates components
  only; rigid-body/joint topology, slot graph, system logic, tuning, and gameplay are unchanged.
- **One atom = one legacy Rapier body.** Every track/wheel/gun is its own physics atom with its
  original joint; carriers are general (a slot's `CarrierRef` → the exact body that owned it).
  `spawnVehicle` rebuilds the full body+joint+slot graph 1:1 — never a hull+turret-only collapse.
- Bodyless rendered children exist & grow → physics/render split with a mirror seam.
- `Joint`/`JointMotor` connect two physics atoms → stay PhysicsWorld-side. No joint references
  a brain/render eid.
- Slots: logical + cold + `#slots ≳ #parts` → own **SlotWorld**.
- Slot↔part: slot is permanent & says *where*; part remembers *which slot* (`HomeSlotRef`);
  detach keeps the home link; heal/upgrade re-registers the link via Bridge.
- Team/Player: canonical in brain (GameWorld) + cheap static copy on the atom for contacts.
- **Bridge is scoped to id synchronization ONLY** (§6): it owns the cross-world link tables +
  translators, nothing else. Spawn/teardown factories, the `mirror-sync` system, attach/detach/heal,
  and the deferred command buffer are SEPARATE concerns that merely call Bridge's `link`/`unlink`/
  translators. Links stored as components, written only by Bridge; debug `validate()` audits the id
  graph.

Remaining:
- **Bullet churn** in PhysicsWorld+RenderWorld — measure eid fragmentation before splitting a
  ProjectileWorld (§9).
- **Command-buffer flush ordering** — exact tick point(s) and whether detach/heal need a separate
  flush phase from spawn/destroy (the buffer is the tick's, not Bridge's).
- **Hot-path audit** — confirm which control/gameplay systems become cross-world and stay
  cold; if any turns hot (dense bullet combat), revisit data placement.
