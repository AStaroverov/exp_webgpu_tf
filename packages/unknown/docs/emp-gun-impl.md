# EMP Gun / EMP Tank — implementation document

A new medium-chassis tank (`VehicleType.EmpTank`) firing a slow, spinning, self-lit
EMP grenade. On detonation it deals token area damage of a new `DamageKind.Emp` and
**fully disables** every vehicle in the blast radius for a fixed duration (`Stunned`):
no track movement, no turret rotation, no firing. The centerpiece is the visuals:
a tumbling glowing grenade, a blue-white electric detonation burst, and crackling
lightning arcs riding the stunned hull for the whole stun.

All paths below are under `packages/unknown/src/Game` unless noted otherwise.
Every cited file, symbol and line was verified against the current source.

---

## 1. Overview & player-facing behavior

- **The weapon.** The EmpTank's turret is a normal `Firearms` gun (NOT
  `StreamFirearms`) firing `BulletCaliber.EmpGrenade`: a slow (280 px/s, vs rocket
  450 and bullets 650+), elongated 12×7 projectile that spins in flight
  (`angularSpeed`, ~2 rev/s) and glows electric blue (its own `LightEmitter` feeds
  the Radiance Cascades pass). Single shot, rocket-class 5000 ms reload.
- **Detonation = contact, no proximity fuse.** Like the rocket: `health: 0.001`
  makes the grenade die on any contact, and the existing `Explodable + Destroy` →
  `createExplodeSystem` pair detonates it uniformly for _every_ death cause —
  contact, `DestroyByDistance` max range (air-burst at the end of the arc),
  out-of-zone. This is literally the CLAUDE.md worked example; the 110 px blast
  radius already buys near-miss forgiveness, so a fuse (sensor collider + arming
  system + collision-group audit) is cut as over-abstraction.
- **The stun.** The blast records an `Emp`-kind hit on every overlapped part; the
  hitable pipeline resolves the part to its vehicle and applies
  `Stunned.refresh(vehicleEid, EmpStunConfig.durationMs)` (2000 ms). Stun is
  binary and **refreshes, never stacks**: `max(remaining, durationMs)`. One blast
  hits N parts of the same vehicle — `max` makes the N applications idempotent;
  a second grenade mid-stun resets to a full 2 s, never beyond.
- **While stunned** the tank cannot move, rotate its turret, or fire — but it is
  still a physics body: explosions, rams and residual momentum push it around.
  That is why the stun visuals are a hull-attached overlay, not world-anchored
  entities (§6).
- **Visual feedback** scales with effect magnitude (project rule): the lightning
  arcs, their ground glow, and the blue part tint all decay with
  `remainingMs / durationMs` and snap back to full on refresh.

---

## 2. New / changed components

Config-key convention (project rule): components store live per-entity state and
config ROW KEYS only — never copies of config values; tunables are read from
`EmpStunConfig` / `EmpVfxConfig` / `BulletCaliberConfig` at the use sites. All
**new** components below follow this. One pre-existing exception this design
builds on: `Explodable` (verified, `ECS/Components/Explodable.ts`) already
stores per-entity _copies_ of the `explosion.{damage, radius, vfxSize,
lightRadius}` tunables from the `BulletCaliberConfig` row, and §2.4 extends
that copied-settings shape with a `kind` column (itself a discriminant key, not
a tunable) rather than refactoring `Explodable` to the rule-conformant shape
(store the caliber/row key, read `explosion` at the explode site). That
refactor is out of scope here; until it happens the blanket rule does not hold
for `Explodable`.

### 2.1 `Stunned` — NEW, `ECS/Components/Stunned.ts`

Mirror of `Slowed` (`ECS/Components/Slowed.ts`), but time-based instead of
magnitude-based. Existence-based state: presence = stunned; no flag, no enum.

| Field         | Type             | Meaning                                |
| ------------- | ---------------- | -------------------------------------- |
| `remainingMs` | `TypedArray.f64` | Live countdown; the only stored state. |

```ts
export const createStunnedComponent = defineComponent((Stunned) => {
  const remainingMs = TypedArray.f64(delegate.defaultSize);
  return {
    remainingMs,
    refresh(world: World, eid: EntityId, durationMs: number) {
      if (!hasComponent(world, eid, Stunned)) {
        addComponent(world, eid, Stunned);
        remainingMs[eid] = 0;
      }
      remainingMs[eid] = Math.max(remainingMs[eid], durationMs);
    },
  };
});
```

`durationMs` comes from `EmpStunConfig.durationMs` at the call site — no copied
tunable. Cap = refresh (`max`), not stack (`+=`); the ceiling is `durationMs`
itself, no separate cap constant.

Register in `ECS/createGameWorld.ts` → `createGameOnlyComponents`, next to
`Slowed: createSlowedComponent(world)` (line 76).

### 2.2 `StunArcs` — NEW, `ECS/Components/StunArcs.ts`

Render-side bookkeeping on the vehicle: which overlay entity carries its arcs.

| Field        | Type             | Meaning                                                                                                  |
| ------------ | ---------------- | -------------------------------------------------------------------------------------------------------- |
| `overlayEid` | `TypedArray.u32` | Eid of the arc-overlay entity; sentinel `0` = none. Relation by eid, verified before use (project rule). |

Register in `createGameWorld.ts` as well. Name says what it stores.

### 2.3 `DamageKind.Emp` — EDIT, `ECS/Components/Damagable.ts`

Current enum is `Physical = 0, Fire = 1, Frost = 2` (verified). Add:

```ts
export enum DamageKind {
  Physical = 0,
  Fire = 1,
  Frost = 2,
  Emp = 3,
}
```

### 2.4 `Explodable` gains `kind` — EDIT, `ECS/Components/Explodable.ts`

Current columns (verified): `damage, radius, vfxSize, lightRadius`. Add:

| Field  | Type            | Meaning                                                                                                                                                                                              |
| ------ | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind` | `TypedArray.i8` | `DamageKind` of the blast's area damage. Discriminant key — visuals are resolved from `ExplosionVisualsByKind[kind]` at the use site; **no `vfxType` column** (that would be a copied config value). |

`ExplodableSettings` += `kind?: DamageKind`; `addComponent` stores
`settings.kind ?? DamageKind.Physical`. Existing rockets stay `Physical`,
behavior unchanged. Not a god-component move: kind is the blast's damage tag,
always co-processed with damage/radius by the one explode system.

### 2.5 `VFXType` additions — EDIT, `ECS/Components/VFX.ts`

`VFXType` is a `const` object (verified, values 0–5). Add:

```ts
EmpArc: 6,    // crackling overlay riding the stunned tank (lives the whole stun)
EmpBurst: 7,  // detonation discharge at the blast epicenter (~450 ms)
```

### 2.6 `BodyOptions.angularSpeed` — EDIT, `Physical/createBody.ts`

`BodyOptions` += `angularSpeed?: number`; the single `RigidBodyDesc` builder chain
(verified at lines 23–31) gains `.setAngvel(o.angularSpeed ?? 0)` next to
`.setLinvel`. Because `CommonRigidOptions = BodyOptions & {…}`
(`Physical/createRigid.ts:18`) and `createRectangleRR` feeds it through, the new
field flows to bullets with **zero signature changes** anywhere else. Declarative
at creation — no post-spawn `setAngvel` round-trip.

---

## 3. Config changes

### 3.1 `Config/weapons.ts`

```ts
export enum BulletCaliber {
  Light = 0,
  Medium = 1,
  Heavy = 2,
  Rocket = 3,
  EmpGrenade = 4,
}
```

`BulletCaliberStats` gains two **optional, presence-driven** fields (same pattern
as the existing `explosion?` — field presence is the trigger, no enum branch
anywhere):

| New field       | Type                                                     | Meaning                      |
| --------------- | -------------------------------------------------------- | ---------------------------- |
| `angularSpeed?` | `number`                                                 | Initial spin, rad/s.         |
| `light?`        | `{ color: [number, number, number]; intensity: number }` | In-flight self-illumination. |

New row (note: `mapBulletCaliber` in `ECS/Components/Bullet.ts` is an alias of
`BulletCaliberConfig`, so the row is picked up everywhere automatically):

```ts
[BulletCaliber.EmpGrenade]: {
    width: 12, height: 7,         // elongated → the tumble reads visually (a circle would hide it)
    speed: 280,                   // SLOW, lobbed feel (rocket 450, bullets 650+)
    density: 5_000,
    damage: 1,                    // token contact damage — the stun IS the payload
    reloadTime: 5_000,            // single shot, rocket-class reload (rocket parity; tune in playtest)
    linearDamping: 0.1,
    maxDistance: HexGridConfig.radius * 4,
    health: 0.001,                // rocket pattern: dies (→ detonates) on any contact
    angularSpeed: Math.PI * 4,    // 2 rev/s; bullet angularDamping 0.1 barely decays it over flight
    light: { color: [0.5, 0.78, 1.0], intensity: 2.5 },
    explosion: { damage: 2, radius: 110, vfxSize: 200, lightRadius: 240, kind: DamageKind.Emp },
},
```

Next to `FrostSlowConfig` (the sibling kind-specialty config, verified at
`weapons.ts:132`):

```ts
export const EmpStunConfig = {
  /** Full vehicle-disable duration applied/refreshed per Emp-kind hit */
  durationMs: 2_000,
} as const;
```

### 3.2 `Config/vfx.ts`

```ts
export const EmpVfxConfig = {
  flash: { color: [0.55, 0.8, 1.0], intensity: 8.0, duration: 450 }, // FlashLightConfig row shape
  arc: { color: [0.55, 0.8, 1.0], lightIntensity: 1.4, lightRadiusPx: 60 },
  burstDurationMs: 450,
  tint: [0.65, 0.8, 1.0] as [number, number, number],
} as const;

/** Detonation visuals keyed by the blast's DamageKind; absent kinds fall back to the
 *  classic fireball + FlashLightConfig.explosion. Data lookup, not a type branch. */
export const ExplosionVisualsByKind: Partial<
  Record<
    DamageKind,
    {
      vfxType: VFXTypeValue;
      durationMs: number;
      flash: { color: readonly [number, number, number]; intensity: number; duration: number };
    }
  >
> = {
  [DamageKind.Emp]: {
    vfxType: VFXType.EmpBurst,
    durationMs: EmpVfxConfig.burstDurationMs,
    flash: EmpVfxConfig.flash,
  },
};
```

(`FlashLightConfig` rows are `{color, intensity, duration}` — verified; the Emp
flash row matches that shape so `spawnLightFlash` consumes it unchanged.)

### 3.3 `Config/vehicles.ts`

```ts
export enum VehicleType { …, FrostTank = 7, EmpTank = 8 }

export const EmpTankConfig: TankStats = {
    ...MediumTankConfig,
    type: VehicleType.EmpTank,
    gun: { ...MediumTankConfig.gun!, caliber: BulletCaliber.EmpGrenade },
};
```

plus a `case VehicleType.EmpTank: return EmpTankConfig;` in `getTankConfig`
(verified switch at lines 393–410). `MediumTankConfig.gun` exists (verified), so
the spread is safe.

### 3.4 `Config/parts.ts`

`VehicleBaseDensity` (verified, lines 48–57) gains:

```ts
[VehicleType.EmpTank]: 30 * 3,   // medium-class
```

---

## 4. Systems

### 4.1 `createStunnedExpirySystem` — NEW, `ECS/Systems/createStunnedExpirySystem.ts`

- **Query:** `[Stunned]`.
- **Responsibility:** `remainingMs[eid] -= delta; if (<= 0) removeComponent(world, eid, Stunned)`.
  Mirror of `createSlowedExpirySystem` but time-based; iterate **backwards**
  (removeComponent swap-removes inside the query's dense array — same comment as
  the Slowed system, verified at `createSlowedExpirySystem.ts:18`).
- **Ordering:** in `gameTick`, immediately after `slowedExpiry(delta)`
  (`createGame.ts:246`) — i.e. after `updateHitableSystem`, so a freshly applied
  stun loses only its first tick's delta, exactly like `Slowed` loses one thaw
  step. Consistent semantics.

### 4.2 `createHitableSystem` — EDIT (`ECS/Systems/createHitableSystem.ts`)

The one sanctioned kind-specialty site. `applyKindEffects` (lines 113–124) already
runs before `saveHitters` / `applyDamage` ("ring readers go first" — order
verified at lines 40–42, unchanged). Restructure its body into a small
`switch (Hitable.getKind(partEid, i))` now that there are two kinds:

```ts
case DamageKind.Frost: { /* existing Slowed.addContribution */ }
case DamageKind.Emp: {
    const vehicleEid = findVehicleEidByPartEid(partEid);
    if (vehicleEid === undefined) break;   // torn-off debris: explicit absence
    Stunned.refresh(world, vehicleEid, EmpStunConfig.durationMs);
}
```

Stun is binary regardless of the tiny `damage * proximity` riding the hit — the
_kind_ triggers it, matching Frost's per-hit semantics. "Adding a kind = enum
value + one branch in applyKindEffects + config row" is satisfied literally.

### 4.3 `createExplodeSystem` — EDIT (`ECS/Systems/createExplodeSystem.ts`)

Stays kind-agnostic in logic; two changes:

- **Area hit kind:** line 65 currently hardcodes `DamageKind.Physical`. Replace:
  `Hitable.hit$(targetEid, eid, damage * proximity, Explodable.kind[eid])`.
  Rockets default to Physical → behavior unchanged. All damage still flows
  through `Hitable.hit$` and is applied only in `applyDamage`.
- **Visuals lookup:**
  ```ts
  const visuals = ExplosionVisualsByKind[Explodable.kind[eid] as DamageKind];
  spawnExplosion({
    x,
    y,
    size: Explodable.vfxSize[eid],
    duration: visuals?.durationMs ?? ExplosionConfig.defaultDuration,
    type: visuals?.vfxType ?? VFXType.Explosion,
  });
  const flash = visuals?.flash ?? FlashLightConfig.explosion;
  spawnLightFlash({
    x,
    y,
    radius: Explodable.lightRadius[eid],
    duration: flash.duration,
    color: flash.color,
    intensity: flash.intensity,
  });
  ```
  `ECS/Entities/Explosion.ts`: `ExplosionOptions` += `type?: VFXTypeValue`
  (default `VFXType.Explosion`), passed to `VFX.addComponent`. No
  `if (caliber === Emp)` anywhere.

### 4.4 `createSpawnerBulletsSystem` — EDIT (`ECS/Systems/createBulletSystem.ts`)

Fire gate (see §5). Query unchanged (`[VehicleTurret, TurretController, Firearms]`).

### 4.5 `createStreamFirearmsSystem` — EDIT

Fire gate (see §5). Query unchanged (`[VehicleTurret, TurretController, StreamFirearms]`).

### 4.6 `createStunArcsSystem` — NEW, `ECS/Systems/Render/createStunArcsSystem.ts`

Early-return `if (!RenderDI.enabled)` (headless training spawns no overlays).
Runs inside **`spawnFrame`**, after `spawnExhaustSmoke` (`createGame.ts:179`) —
structural adds stay at the sanctioned phase boundary.

- **Query A:** `[Stunned, Vehicle, Not(StunArcs)]` → spawn the overlay. Iterate
  **backwards**: the final `StunArcs.addComponent` swap-removes the current
  vehicle from this query's live dense array (the `Not(StunArcs)` term), so a
  forward loop would skip the swapped-in entity whenever two or more vehicles
  are stunned the same frame — same rule as the `createSlowedExpirySystem`
  comment (verified, `createSlowedExpirySystem.ts:18`) and the tint revert pass
  in `createTintSystem.ts`. Per vehicle:
  `addEntity` + `addTransformComponents` (identity `LocalTransform`) +
  `Parent.addComponent(world, overlayEid, vehicleEid)` **and**
  `Children.addChildren(vehicleEid, overlayEid)` — `Parent.addComponent` only
  writes `Parent.id` (verified, `ECS/Components/Parent.ts`); registering the
  overlay in the vehicle's `Children` (the `ECS/Entities/Sound.ts:59` precedent)
  is what lets `recursiveTypicalRemoveEntity` reach the overlay when the vehicle
  dies (§10 Q6);
  `VFX.addComponent(world, overlayEid, VFXType.EmpArc)`;
  `Progress.addComponent(world, overlayEid, EmpStunConfig.durationMs)`;
  `Shape.addComponent(world, overlayEid, ShapeKind.Circle, EmpVfxConfig.arc.lightRadiusPx)`
  - `Color.addComponent(world, overlayEid, ...EmpVfxConfig.arc.color, 0)`
  - `LightEmitter.addComponent(world, overlayEid, EmpVfxConfig.arc.lightIntensity)`
    (the `spawnLightFlash` alpha-0-circle precedent, verified in
    `ECS/Entities/LightFlash.ts`); finally
    `StunArcs.addComponent(world, vehicleEid, overlayEid)`.
- **Query B:** `[StunArcs]` — iterate **backwards** too: the expired branch's
  `removeComponent(world, vehicleEid, StunArcs)` swap-removes mid-loop, same as
  above —
  - vehicle still `Stunned` and overlay alive → **rewind the shader clock**:
    `Progress.age[overlayEid] = Progress.maxAge[overlayEid] - Stunned.remainingMs[vehicleEid]`
    (a stun refresh rewinds progress so the shader's tail fade always tracks true
    remaining time), and flicker the ground glow:
    `LightEmitter.set$(overlayEid, base * frac * (0.6 + 0.8 * sliceHash), 0)`
    where `frac = remainingMs / durationMs` (magnitude-decaying base) and
    `sliceHash` mirrors the shader's 24-slice hash in cheap JS so glow strobes in
    sync with the bolts.
  - else (stun expired) → verify the overlay eid still exists,
    `scheduleRemoveEntity(overlayEid)` (`ECS/Utils/typicalRemoveEntity.ts` —
    adds `Destroy`, consumed by `destroyFrame`), and
    `removeComponent(world, vehicleEid, StunArcs)`.
  - **Vehicle death is NOT handled here** — a removed vehicle drops out of the
    `[StunArcs]` query, so no "vehicle dead" branch in this system can ever
    fire. That path is covered by the `Children.addChildren` registration in
    Query A: the vehicle's recursive destroy
    (`recursiveTypicalRemoveEntity` walks `Children`,
    `ECS/Utils/typicalRemoveEntity.ts:25–31`) takes the overlay down with it.
    The existence check before `scheduleRemoveEntity` above is what makes the
    race benign when stun expiry and vehicle death land the same frame (§10 Q6).

**Why a Parent-attached overlay works:** an entity with `Parent` +
`Local/GlobalTransform` and **no** `RigidBodyRef` is recomputed by
`createAttachedTransformSystem` (`global = parentGlobal * local`) right after the
rigid-body sync (verified: `createAttachedTransformSystem.ts`, called at
`createGame.ts:127`). The overlay therefore follows and rotates with the hull
even when the disabled tank is _pushed_ by physics — one persistent entity, zero
per-frame spawns.

### 4.7 `createTintSystem` — EDIT (`ECS/Systems/Render/createTintSystem.ts`)

Third tint pass between the Fire-Dot pass and the Frost pass: query
`[Stunned, Children]`, walk slot parts with the existing recursive walker.
**Parameterize `frostTintSlotParts(parentEid, intensity)` →
`tintSlotParts(parentEid, tint, intensity)`** — this is the walker's second
mechanism (the Fire pass is per-part and doesn't use it), so this is
parameterization of an existing helper, not new abstraction.

- Intensity = `Stunned.remainingMs[vehicleEid] / EmpStunConfig.durationMs`
  (magnitude-scaled lerp from `OriginalColor`, per project rule), tint
  `EmpVfxConfig.tint`.
- Precedence per part: **Fire > Emp > Frost** (fire is damage — keep visible;
  EMP outranks frost because the tank is fully disabled). Implement as the
  existing `if (isFireDot(partEid)) continue;` plus an `isEmpStunned` skip in
  the frost pass.
- The revert pass (lines 78–89) gains one more "still tinted?" check: keep the
  `OriginalColor` while the part's vehicle has `Stunned`.

### 4.8 Exact ordering in `createGame.ts` (`gameTick`, deterministic, load-bearing)

```
spawnFrame:     spawnBullets → streamEmit → spawnTreadMarks → spawnWheelTreadMarks
                → spawnExhaustSmoke → spawnStunArcs                       (NEW, last)
physicalFrame:  updateTrackControl (stun ×0 factor) → updateWheelControl
                → updateTurretRotation (stun ×0; setTargetPosition$ STILL called)
                → … → physicalWorld.step → syncRigidBodyState
                → applyRigidBodyDeltaToLocalTransform → updateAttachedTransforms
                                                          (overlay follows hull here)
                → event drains (drainContactForceEvents / drainCollisionEvents —
                  AFTER the transform sync; hits still land before updateHitableSystem)
updateGridOccupancy
updateActions            (untouched — gates live at the spawners, §5)
applySensorHits
plugins[Before]
updateHitableSystem      (applyKindEffects: Emp → Stunned.refresh — ring readers
                          before ring reset; before saveHitters/applyDamage, order unchanged)
updateTankAliveSystem
regenerateShields
dotTick
slowedExpiry
stunnedExpiry            (NEW — immediately after slowedExpiry; after updateHitableSystem
                          so a fresh 2000 ms stun survives its first tick minus one delta,
                          like Slowed loses one thaw step)
visTracksUpdate
statusTint               (reads Stunned the same frame it was applied — no one-frame tint lag)
updateProgress           (advances the overlay's arc animation; the StunArcs age-rewind
                          runs in spawnFrame at the top of the tick, so a refresh landing
                          later the same tick in updateHitableSystem is rewound only next
                          tick — the same accepted one-frame semantics as the stun gates
                          and the overlay spawn, not zero lag)
lightFade → updateTreadMarks → camera
destroyFrame:   destroyByTimeout → destroyByDistance → destroyOutOfZone
                → explode (reads Explodable.kind → EmpBurst + blue flash) → destroy
                                                          (existing pair, untouched)
renderFrame / soundFrame / plugins[After]
```

Stun lands in `updateHitableSystem` and is consumed by the physicalFrame gates
**next** tick — the same one-tick semantics as `Slowed`, accepted and consistent.
`spawnStunArcs` in `spawnFrame` means the overlay appears one tick after the stun
(imperceptible at 60 fps) and its `GlobalTransform` resolves the same frame
(spawn-at-top-of-tick rationale, see the comment at `createGame.ts:225–229`).

---

## 5. Stun gating — exact files & sites

Four edits, all `hasComponent` membership checks; zero weapon-enum branches.

1. **Movement — `ECS/Systems/Vehicle/TrackControlSystem.ts:46`** (verified: the
   `slow` factor lives there, applied at line 73):

   ```ts
   const stun = hasComponent(world, vehicleEid, Stunned) ? 0 : 1;
   …
   applyTrackImpulse(childEid, power * impulseFactor * slow * stun, vehicleRotation, delta);
   ```

   Multiply, not `continue` — keeps the loop shape uniform with `slow`, and the
   existing `if (impulseFactor === 0) return;` early-out in `applyTrackImpulse`
   (line 28) skips the `Impulse.add` accumulation anyway (the accumulator is
   consumed later by `createApplyImpulseSystem`; Rapier is only reached there,
   not in `applyTrackImpulse` itself).

2. **Turret rotation — `ECS/Systems/Vehicle/VehicleControllerSystems.ts:19–24`**
   (verified): multiply the same `stun` factor into `deltaRot` at line 22, next
   to the existing `slow`:

   ```ts
   const deltaRot = turretRotDir * maxRotationSpeed * slow * stun * (delta / 1000);
   JointMotor.setTargetPosition$(turretEid, relTurretRot + deltaRot);
   ```

   **Do NOT `continue`:** `setTargetPosition$` must still run every frame so the
   joint motor actively _holds_ the turret pinned at `relTurretRot + 0` instead
   of drifting on a stale target.

3. **Bullet firing — `ECS/Systems/createBulletSystem.ts`**, in the turret loop of
   `createSpawnerBulletsSystem`, after `Firearms.updateReloading` and before the
   shoot-flag consumption (line 18):

   ```ts
   if (hasComponent(world, Parent.id[turretEid], Stunned)) continue;
   ```

   Reload keeps ticking above the gate — stun doesn't refund or pause reload.

4. **Stream firing — `ECS/Systems/createStreamFirearmsSystem.ts`**, identical
   guard on `Parent.id[turretEid]` before the emit accumulator/`emitBurst` path
   (covers FlameTank/FrostTank if they're ever stunned).

**Deliberately NOT gated:** `ECS/Actions/systems/FireAction.ts` /
`FireStreamAction.ts` (the decision layer). Gating at the two spawner/emitter
systems covers every driver uniformly — policy actions, demo driver, debug
manual control — and is self-healing: a stunned tank's Fire action raises the
shoot flag, the suppressed spawner never consumes it, the action sits in FIRING,
and the round fires the instant the stun lifts. The ActionScheduler watchdog
(`MAX_ACTION_MS = 5000`, `ECS/Actions/ActionSlot.ts:21`) covers pathological
stalls — see open question Q4. `WheelControlSystem` (MeleeCar) is out of scope —
same one-liner if EMP should disable wheeled vehicles (Q5).

---

## 6. VFX & lighting

### 6.1 Technique choice

Both new effects are **fragment-pattern instanced-quad VFX** rendered entirely in
`ECS/Systems/Render/VFX/vfx.shader.ts` — the purpose-built home for animated
transient effects (Flame/Frost precedent): instanced quads, per-instance
`(progress, seed, type, maxRadius)`, all animation in the fragment shader,
fixed-bound loops the compiler unrolls, zero CPU geometry, zero new pipelines.

A new lightning `ShapeKind` SDF was **rejected**: the SDF path
(`packages/renderer/src/ECS/Systems/SDFSystem/sdf.shader.ts`) is the _lit world
geometry_ path — every shape feeds the shadow-map, emission and JFA passes, and a
new kind costs ~6 touch-points across renderer files. A flickering re-hashed
polyline would need per-frame CPU vertex updates and would pollute the
shadow/JFA passes with transient geometry.

A bolt is _distance-to-a-noise-displaced chord_: `|q.y − offset(u, t, seed)|` in
bolt-local coordinates, with a `sin(π·u)` envelope pinning the endpoints, plus
**time-quantized re-strike flicker** (`slice = floor(progress * 24.0)` re-hashes
endpoints and gates brightness per slice) — the quantization is what makes it
read as lightning rather than a wavy line.

### 6.2 WGSL — `ECS/Systems/Render/VFX/vfx.shader.ts`

- `WGSL_CONSTANTS` (line 12) += `const VFX_EMP_ARC: u32 = 6u; const VFX_EMP_BURST: u32 = 7u;`
- New `WGSL_EMP` block, concatenated into the shader template after `WGSL_FROST`
  (line 653); two new cases in the `fs_main` switch (line 714).
- **Available utils are the shader-local ones** (`WGSL_UTILS`, lines 57–93):
  `hash21`, `hash22`, `noise2D`, `fbm(p, octaves)`. ⚠️ `valueNoise` / `rotate2d`
  live in `ECS/Systems/Render/noise.wgsl.ts`, which is **not** part of this
  shader's template — use `noise2D` for the high-frequency jitter octave (it is
  the same Hermite-interpolated value noise).

Functions:

- **`empBolt(p, a, b, t, seed, amp) -> f32`** — distance to a jagged bolt between
  endpoints `a`,`b`: project onto the chord for bolt-local `(u, q.y)`; envelope
  `env = sin(π·u)` pins endpoints; displacement = two octaves —
  `fbm(vec2f(u * 6.0 + seed * 17.0, t * 3.0), 2)` low-freq writhe +
  `noise2D(vec2f(u * 22.0 + seed * 31.0, t * 9.0))` high-freq jitter — scaled by
  `amp * env`.
- **`renderEmpArcs(p, progress, seed) -> vec4f`** — 3 chords across the hull disc
  (endpoints on a radius-`0.8R` circle, re-hashed per slice);
  `slice = floor(progress * 24.0)` → 12 re-strikes/s over a 2 s stun; per-bolt
  per-slice strobe gate `step(0.35, hash21(vec2f(slice, seed + boltIndex)))`
  (~35% of slices dark); color = `smoothstep(1.6, 0.0, d)` white core +
  `exp(-d * 0.12)` electric-blue halo `vec3f(0.35, 0.6, 1.0)`; alpha
  `*= 1.0 - smoothstep(0.85, 1.0, progress)` tail fade over the last ~300 ms.
  Because `Progress` is rewound on refresh (§4.6), `progress` tracks true
  elapsed/remaining stun **up to the same accepted one-frame lag as everything
  else in this design** (§4.8): a refresh lands in `updateHitableSystem`
  mid-tick, but the rewind runs in `spawnFrame` at the top of the _next_ tick
  (and reads `remainingMs` before that tick's `stunnedExpiry` decrement, so it
  is one delta stale). The §1 "snap back to full brightness" therefore happens
  one tick after the refreshing hit — imperceptible at 60 fps —
  **arc brightness decays with effect magnitude**.
- **`renderEmpBurst(p, progress, seed) -> vec4f`** — three layers:
  (1) expanding ring shockwave `smoothstep(0.07, 0.0, abs(r - ease))` with an
  easeOutQuad front; (2) 5 radial lightning branches with constant-pixel width
  (angular distance × `r·R` converts to arc-length px), fbm-wobbled, trailing
  the ring front via `smoothstep(ease + 0.1, ease - 0.3, r)`, re-rolled per
  `slice = floor(progress * 10.0)`; (3) central white-blue flash
  `exp(-r * 6.0) * pow(1.0 - progress, 2.0)`. All loops fixed-bound (3 bolts /
  5 branches).

### 6.3 `createDrawVFXSystem.ts` maps (lines 13–44, verified)

Both `getMaxRadius` and `seedMultiplier` are `Record<VFXTypeValue, …>` — they
**must** be extended or the new types index `undefined`:

```ts
// getMaxRadius — like Flame/Frost, the transform scale is 1 (the overlay rides
// Parent-attached transforms), so the world-pixel radius lives here:
[VFXType.EmpArc]:   () => 70,                        // ≈ medium-hull circumradius + margin
[VFXType.EmpBurst]: (p) => 200 * (0.55 + 0.45 * p),  // matches the shader's R

// seedMultiplier
[VFXType.EmpArc]: 0.157,
[VFXType.EmpBurst]: 0.191,
```

### 6.4 Lighting (Radiance Cascades — all reuse, zero RC changes)

All light rides the existing alpha-0-circle + `LightEmitter` mechanism (emitters
feed the RC emission pass and skip baked shadow darkening; `spawnLightFlash` and
stream-particle precedents verified):

- **Flying grenade** — in `createBullet` (`ECS/Entities/Bullet.ts`):
  ```ts
  if (bulletCaliber.light && RenderDI.enabled) {
    Color.set$(bulletId, ...bulletCaliber.light.color, 1);
    LightEmitter.addComponent(world, bulletId, bulletCaliber.light.intensity);
  }
  ```
  The RC emission pass uses the entity's own spinning 12×7 SDF rect as the
  emitter silhouette, so the tumble gives a subtly strobing glow for free.
  Ordering note: `spawnBullet` computes its dimmed vehicle color _before_ calling
  `createBullet` (verified, `Bullet.ts:110–124`), so the `Color.set$` override
  here wins — the glow color is the final color. `RenderDI.enabled` guard matches
  the stream-particle precedent (`createStreamFirearmsSystem.ts:104`).
- **Detonation flash** — existing `spawnLightFlash` (internally render-guarded),
  flash row selected via `ExplosionVisualsByKind` (§4.3): cold blue-white,
  intensity 8, 450 ms, at `Explodable.lightRadius` (240).
- **Stun overlay light** — the overlay entity's `Shape.Circle(60)` +
  `Color(arc.color, a = 0)` + `LightEmitter`. Per-frame in
  `createStunArcsSystem`: `LightEmitter.set$(overlayEid, intensity, 0)` with
  `intensity = arc.lightIntensity * frac * (0.6 + 0.8 * sliceHash)` — ground glow
  strobes in sync with the bolts (same 24-slice hash) and dims as the stun
  expires (`frac = remainingMs / durationMs`). `LightEmitter.set$(eid, i, r)`
  signature verified in `packages/renderer/src/ECS/Components/Common.ts:77`.

---

## 7. Entity factory & VehicleType wiring

- **`ECS/Entities/Tank/Medium/EmpTank.ts`** (NEW): `createMediumTank` sibling on
  the medium chassis (`MediumTankParts.ts` constants), built with `createTankBase`
  → `createTankTracks` → **`createTankTurret`** (normal `Firearms`, verified
  `Common/Tank.ts:61–71` — NOT `createStreamTankTurret`). Set
  `options.firearms.bulletCaliber = getTankConfig(VehicleType.EmpTank).gun!.caliber`
  — values from config, not literals (note: `createMediumTank` currently
  hardcodes `BulletCaliber.Medium` at `MediumTank.ts:71`; the new factory should
  follow the config convention, and the medium factory can be migrated
  opportunistically). Optional blue palette accent on the turret.
- **`ECS/Entities/Tank/createTank.ts`**: add
  `case VehicleType.EmpTank: return createEmpTank(options);` to the switch and
  the `TankVehicleType` union / `TankOptions` union.
- **`ECS/Entities/Vehicle/VehicleBase.ts`**: `volumeByType[VehicleType.EmpTank] = 0.8`
  (medium-class; the Record is exhaustive over `VehicleType` — verified — so this
  is a compile error if forgotten).
- **`setupDemoWorld.ts`**: the demo currently pins the _last_ slot to RocketTank
  (verified). Pin the **second-to-last** slot to EmpTank the same way (or bump
  `TANK_COUNT` to 6) — the visuals must be inspectable in dev.
- **`packages/ppo_unknown/src/env/createUnknownScenario.ts`**: `TANK_TYPES`
  (verified lines 46–53: Light, Medium, Heavy, Rocket, Flame, Frost) +=
  `VehicleType.EmpTank`.
- Loadout is decided entirely by the factory + vehicle-type pool; downstream code
  keys off component presence (`Firearms`) and `DamageKind.Emp` — no weapon-enum
  branches, no scenario-level component swapping.

---

## 8. Collision groups checklist

**No `Config/physics.ts` edits needed** — stated explicitly so the bidirectional
grep rule is visibly satisfied:

- [x] The grenade is a regular bullet: `belongsCollisionGroup = CollisionGroup.BULLET`,
      `interacts = ALL & ~TANK_TURRET_GUN_PARTS & ~BULLET` (verified,
      `ECS/Entities/Bullet.ts:39–40`).
- [x] `BULLET` ↔ `VEHICLE_HULL_PARTS` / `TANK_TURRET_HEAD_PARTS` / `OBSTACLE`
      masks are already bidirectional (existing bullets work).
- [x] The blast itself is not a collider — `createExplodeSystem` does a distance
      query over `[Hitable, Not(Bullet)]`, no sensor involved.
- [x] The arc overlay and burst VFX entities have **no** rigid body at all.
- [x] No new `CollisionGroup` value is introduced anywhere in this feature.

---

## 9. Implementation order (step-by-step PR plan)

Each step compiles and is independently testable.

1. **Plumbing — kinds & spin** (headless-safe):
   `DamageKind.Emp`; `Explodable.kind` column + `ExplodableSettings.kind?`;
   `BodyOptions.angularSpeed` + `.setAngvel` in `createBody`;
   `BulletCaliberStats.angularSpeed?/light?`.
2. **Config rows:** `BulletCaliber.EmpGrenade` + its `BulletCaliberConfig` row;
   `EmpStunConfig`; `EmpVfxConfig` + `ExplosionVisualsByKind`;
   `VehicleType.EmpTank` + `EmpTankConfig` + `getTankConfig` case;
   `VehicleBaseDensity` row.
3. **Projectile:** `createBullet` reads `angularSpeed`/`light` from the caliber
   row, passes `explosion.kind` through. `createExplodeSystem` kind passthrough +
   visuals lookup; `ExplosionOptions.type?`. _Test:_ spawn an EmpTank via console,
   grenade tumbles, glows, detonates with the classic fireball (shader not yet in).
4. **Stun gameplay:** `Stunned` component (+ registration);
   `createStunnedExpirySystem` (+ wiring after `slowedExpiry`); the
   `applyKindEffects` switch + Emp case; the four gates of §5.
   _Test (headless-capable):_ a stunned tank ignores move/rotate/fire inputs for
   2 s, reload still ticks, refresh-not-stack under double hits.
5. **Tank factory:** `EmpTank.ts`, `createTank` case, `volumeByType`,
   `setupDemoWorld` slot. _Test:_ demo world always shows an EmpTank.
6. **Shaders:** `VFXType.EmpArc/EmpBurst`; `WGSL_EMP` (constants, `empBolt`,
   `renderEmpArcs`, `renderEmpBurst`, template concat, switch cases);
   `getMaxRadius`/`seedMultiplier` entries. _Test:_ detonation shows the EmpBurst.
7. **Stun overlay:** `StunArcs` component (+ registration);
   `createStunArcsSystem` wired into `spawnFrame`; Progress rewind + flickering
   `LightEmitter`. _Test:_ arcs ride a stunned tank being pushed by an explosion;
   refresh restores full brightness; overlay dies with the stun and with the tank.
8. **Tint:** `tintSlotParts` parameterization, Emp pass, Fire > Emp > Frost
   precedence, revert-pass check.
9. **Training pool:** `TANK_TYPES` += EmpTank (`ppo_unknown`). Keep this last —
   it changes training distribution (see Q1 before enabling for real runs).

---

## 10. Open questions / tuning notes

1. **RL observation channel** — one dense scalar `remainingMs / durationMs`
   (default 0) is the natural "few and dense" encoding; nothing existing implies
   it (unlike Chilled→`SlowMul`). But it changes the observation shape →
   retraining. Defer to a separate change until EmpTank enters serious training.
2. **Sound** — no electric-zap `SoundType` exists; add `SoundType.EmpZap` at
   detonation + a crackle loop while stunned, or ship silent first?
3. **Friendly fire / self-stun** — the blast stuns every vehicle in radius
   including allies/self (`createExplodeSystem` doesn't team-filter — matches
   existing explosion semantics). Self-stun at minimum range is a real and
   arguably desirable risk of a slow lobbed grenade — deliberate choice, confirm.
4. **Action watchdog vs stun** — `MAX_ACTION_MS = 5000` (verified). A stunned
   tank's Fire front can wait reload (≤5000) + stun (≤2000) ≈ up to ~7 s in the
   worst stacking — the watchdog will time it out and requeue, which is
   acceptable (timeout-requeue is its job), but confirm the requeue path doesn't
   double-spend the reload; bump the constant if FIRING churn shows up.
5. **Wheeled vehicles** — `WheelControlSystem` (MeleeCar) not gated; same
   one-line `stun` factor if EMP should disable wheels.
6. **Overlay cleanup on vehicle death** — note the original sketch had a hole,
   not a double-removal: `Parent.addComponent` alone does not register the
   overlay in the vehicle's `Children` (verified, `ECS/Components/Parent.ts`
   only writes `Parent.id` — the `ECS/Entities/Sound.ts:59` precedent calls
   `Children.addChildren` explicitly), so `recursiveTypicalRemoveEntity` (which
   walks `Children`) could never reach the overlay; and once the vehicle entity
   is removed it drops out of the `[StunArcs]` query, so the system's expiry
   branch could never fire either — the overlay would be removed _zero_ times,
   left orphaned with a dangling `Parent.id` read every frame by
   `createAttachedTransformSystem`. Fixed in §4.6 Query A: the overlay is
   registered via `Children.addChildren(vehicleEid, overlayEid)`, making the
   vehicle's recursive destroy the death-path cleanup. The residual concern is
   the inverse: stun expiry and vehicle death landing the same frame must not
   remove the overlay twice — the existence check before `scheduleRemoveEntity`
   (and `Destroy` being idempotent) covers it.
7. **Per-vehicle arc radius** — fixed 70 px quad fits the medium chassis; if
   heavies/harvesters ever get stunned visibly wrong-sized, bake per-vehicle
   scale into the overlay's `LocalTransform` — don't add a config channel until
   it matters.
8. **Stun vs Slowed interaction** — independent (stun gates to zero anyway;
   `Slowed` keeps thawing underneath). Confirm no need to pause/clear Slowed
   during stun.
9. **Spin direction** — fixed sign from config; randomize per shot
   (`bulletId & 1 ? +1 : -1`) if uniform spin looks robotic.
10. **Reload 5000 ms** — set at rocket parity as the playtest starting point (the
    originally proposed 6000 would be the slowest in the game and untested).
    Tune against how punishing a 2 s full disable feels.

---

## File change inventory

**New (5):** `ECS/Components/Stunned.ts`, `ECS/Components/StunArcs.ts`,
`ECS/Systems/createStunnedExpirySystem.ts`,
`ECS/Systems/Render/createStunArcsSystem.ts`,
`ECS/Entities/Tank/Medium/EmpTank.ts`.

**Edited:** `Config/weapons.ts`, `Config/vfx.ts`, `Config/vehicles.ts`,
`Config/parts.ts`, `Physical/createBody.ts`, `ECS/Components/Damagable.ts`,
`ECS/Components/Explodable.ts`, `ECS/Components/VFX.ts`,
`ECS/createGameWorld.ts`, `ECS/Entities/Bullet.ts`, `ECS/Entities/Explosion.ts`,
`ECS/Entities/Tank/createTank.ts`, `ECS/Entities/Vehicle/VehicleBase.ts`,
`ECS/Systems/createHitableSystem.ts`, `ECS/Systems/createExplodeSystem.ts`,
`ECS/Systems/createBulletSystem.ts`, `ECS/Systems/createStreamFirearmsSystem.ts`,
`ECS/Systems/Vehicle/TrackControlSystem.ts`,
`ECS/Systems/Vehicle/VehicleControllerSystems.ts`,
`ECS/Systems/Render/createTintSystem.ts`,
`ECS/Systems/Render/VFX/vfx.shader.ts`,
`ECS/Systems/Render/VFX/createDrawVFXSystem.ts`, `createGame.ts`,
`setupDemoWorld.ts`, `packages/ppo_unknown/src/env/createUnknownScenario.ts`.
