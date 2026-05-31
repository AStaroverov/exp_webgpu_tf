# Unknown — game prototype

> Prototype of a new game (working name `unknown`). Desert / "dune" setting:
> sand, spice, harvesters, sandstorms, desert flora. Built on the same
> ECS + WebGPU + Rapier foundation as the `tanks` package, but it is a separate
> clean prototype into which only the needed mechanics are ported.

## Running

```bash
npm run dev      # vite dev server (config.vite.ts), HMR disabled
npm run build    # tsc + vite build
npm run preview
```

Entry point: `index.html` → `index.ts` → `src/Game/createGame.ts`.
`index.ts` creates the canvas, calls `createGame({width, height})`,
`game.setRenderTarget(canvas)`, and runs a `requestAnimationFrame` loop calling
`game.gameTick(delta)` (delta is clamped to 16.6667 ms).
Sound is enabled via a button (browser user-gesture requirement) → `game.enableSound()`.

## Stack

- **ECS**: [`bitecs`](https://github.com/NateTheGreatt/biteCS) `^0.4.0`
- **Physics**: `@dimforge/rapier2d-simd` (2D, WASM)
- **Rendering**: WebGPU via the local `packages/renderer` package (SDF shapes, WGSL shaders)
- **Reactivity**: `rxjs`, plus the `delegate`/`obs` helpers for components
- **Hex grid**: `honeycomb-grid` `^4.1.5` (geometry, neighbors, distance)
- **Build**: Vite 6 + `vite-plugin-wasm` + `vite-plugin-top-level-await`

## Architecture

### Dependency on `packages/renderer`
The prototype imports the rendering base directly via relative paths
(`../../../renderer/src/...`): WebGPU init (`gpu.ts`), the frame loop
(`WGSL/createFrame.ts`), the `TransformSystem`, `SDFSystem/createDrawShapeSystem`,
`ChangedDetectorSystem`, `ResizeSystem` systems, and the base render components
(`Color`, `Shape`, `LocalTransform`/`GlobalTransform`, `Roundness`, `Thinness`, `Rope`).

### ECS world — `src/Game/ECS/createGameWorld.ts`
`createGameWorld()` builds a `bitecs` world with a `{ components, time }` context.
`GameComponents = RenderComponents (from renderer) + game components`.
Game components live in `ECS/Components/*`, created via the `defineComponent`
factory (renderer) using the `obs(...)`/`TypedArray` pattern for reactive fields.

### DI singletons — `src/Game/DI/`
State is kept in module-level objects (not classes):
- **`GameDI`** — the main game object. Holds `world`, `physicalWorld`, `width/height`
  and the public methods: `gameTick`, `destroy`, `enableSound`, `setRenderTarget`,
  `setCameraTarget`, plus `plugins`. `createGame()` returns `GameDI` itself.
- **`RenderDI`** — GPU device/context/canvas, `renderFrame`, `destroy`. Rendering is
  optional (the game can run without rendering — for future headless/ML use).
- **`SoundDI`** — sound state, `soundFrame`, `destroy`. Enabled lazily.
- **`PluginDI`** — extensibility: `addSystem(group, system, dispose)`,
  where `group ∈ SystemGroup.{Before, After}` (`ECS/Plugins/systems.ts`).
  These systems run inside `gameTick` before/after the core logic.
- **`PlayerEnvDI`** — the current player's tankEid.

### Frame — `GameDI.gameTick(delta)`
Order (see `createGame.ts`):
1. `physicalFrame` — control (track/wheel/turret) → transform → impulses →
   `physicalWorld.step(eventQueue)` → rigid body sync → contact-force events
   (deals damage via `Hitable.hit$`).
2. `plugins[Before]`
3. Gameplay: hitable, tank-alive, shield regeneration, tracks, progress, tread marks.
4. Camera → `setCameraPosition`.
5. `destroyFrame` (timeout / speed / out-of-zone / destroy) and `spawnFrame`
   (bullets, tread marks, exhaust).
6. `RenderDI.renderFrame?` + `SoundDI.soundFrame?`
7. `plugins[After]`

### Entities — `src/Game/ECS/Entities/`
Composition "vehicle = base + slots + parts" (`Vehicle/`, `Slot`, `VehiclePart`).
Vehicle types are in `Config/vehicles.ts` (`VehicleType`):
`LightTank, MediumTank, HeavyTank, PlayerTank, Harvester, MeleeCar`.
Tanks: `Entities/Tank/{Common,Light,Medium,Heavy,Player}`. Also `Harvester`, `MeleeCar`,
`Wheel`, `Track`. Effects: `Explosion, MuzzleFlash, HitFlash, ExhaustSmoke, TreadMark`.
`GameSession`, `GameMap`, `Player`, `Sound`.

### Render systems — `ECS/Systems/Render/`
SDF shapes (via renderer) + custom WGSL passes: `Fauna` (desert flora),
`Grass`, `VFX`, `Grid` (hex grid overlay), post-effects `Sandstorm` and `Pixelate`.
Background is a sand color.

Frame draw order (in `createGame` `frameTick`): `fauna → grid → shapes → vfx → sandstorm`.
Note: the SDF shape pass writes clip-space `z = 0` for everything, so **layering is by
draw order**, not depth — the grid is drawn after fauna and before shapes so it sits
under the tanks. The `Grid` pass (`Render/Grid/`) is instanced (one quad per hex) and
draws a pointy-hex SDF outline + faint fill; cell centers come from `MapDI.grid`.

### ActionSchedule — `src/Game/ECS/Actions/`
A step-by-step action system in its **own ECS world** (`createActionWorld` →
`ActionScheduleDI.world`), kept separate from the game world; an action only
*references* game entities by id (`ownerEid`, an Entity target's `eid`) — disjoint
id spaces. **One global FIFO stack** of action entities (`ActionScheduleDI.stack`),
played out **chess-like**: only the **top** action is active at any moment. Each
action has a `status` (`Idle/Running/Finished`), a `kind` (`ActionKind`), an owner,
an `ActionTarget` (entity/hex/point addressing — Shape-style `kind`+`values`), and a
per-kind params component. Each kind is one **descriptor** (`ActionDescriptor<Spec>`,
co-located with its system) that forms its own action entity; `ACTION_REGISTRY` maps
kind → descriptor and `EnqueueActionSpec` is *derived* from it. Split of responsibility:
- **Executor systems** (`systems/create<Kind>ActionSystem.ts`, one per kind, wired via
  `registry.ts` → `createRunExecutors`) read the action world for the action and the
  game world for the owner, decide if the top is theirs, and drive it
  `Idle → Running → Finished`, mutating the action as it progresses.
- **Scheduler** (`createActionSchedulerSystem`) owns the stack only: it reaps the
  `Finished` top and deletes the entity. Runs after the executors each tick (in
  `gameTick`, the `updateActions(delta)` block, before the gameplay systems).

Enqueue with `enqueueAction(ownerEid, { kind, target, params })`. Kinds so far:
`MoveToHex` (A* + `VehicleController` to a hex center, updates occupancy), `Wait`
(timer), `TurretAim` (rotate the tank turret toward a target via `TurretController`,
finishes within `tolerance`), `Fire` (raise `TurretController.shoot` and let the
bullet spawner fire `shots` rounds, one per reload cycle). See `Actions/PLAN.md`.

### Config — `src/Game/Config/` (re-exported via `Config/index.ts`)
`vehicles, weapons, parts, obstacles, physics, gameplay, sound, vfx, spice, zindex`.
Physics: collision groups (bitmasks), damping, movement impulses (`physics.ts`).

### `GameMap` — `ECS/Entities/GameMap.ts`
Holds the world offset for an "infinite" map and world↔view coordinate conversion.
Currently the offset is static at the screen center.

### Hex grid — `src/Game/Map/`
The walkable grid units move on. Built on `honeycomb-grid` (pointy-top, rectangular
`cols × rows`), kept **separate from the game ECS world** — occupancy is stored by
reference so it can be wired to ECS later.
- **`HexConfig.ts`** — grid config (hex `radius`, `cols/rows`, pointy orientation),
  the `HexTile` class (`defineHex`), and `POINTY_DIRECTIONS` (the 6 valid neighbor
  directions: E/W + 4 diagonals).
- **`HexGrid.ts`** — wraps the honeycomb `Grid` plus a per-cell store. Each `HexCell`
  has `walkable` + an occupant reference `(occupantEid, occupantWorldId)`.
  `MapWorldId` distinguishes which world an occupant id belongs to (entity id spaces
  are disjoint between worlds). Helpers: `hexToWorld`/`worldToHex`/`cornersOf`,
  `neighbors`, `distance`, `isPassable`, `occupy`/`vacate`/`getOccupant`. The grid is
  centered on the world via the `center` constructor option (origin offset).
- **`findPath.ts`** — A* over the grid (honeycomb has no pathfinder). Heuristic =
  exact hex distance (admissible → optimal); blocked cells are skipped dynamically
  via `isPassable`/a custom `isBlocked`, so no graph rebuild on occupancy change.
  Binary-heap open set. Verified: detours around walls, returns `null` when blocked.
- **`MapDI.ts`** — singleton `{ grid }`, set in `createGame`, cleared on destroy.

## Current state (as of 2026-05-30)

- Working prototype: `createGame()` spawns 4 demo tanks (`spawnDemoTanks`) in a circle;
  they drive forward, rotate, and shoot; the camera follows the first one.
- Full physics → render → sound pipeline works.
- `createMapSystem.ts` is **fully commented out** (the old tile/matrix map generation
  from the `tanks` package) — there is no map yet.
- ML/training from `tanks`/`ppo` is **not ported** here (comment mentions are legacy).

## Goals / Roadmap

### 🎯 Goal 1 (current): Hex grid for unit movement
Build a hexagonal grid (hex grid) for units to move across.

Decisions: pointy-top, rectangular `cols × rows`, `honeycomb-grid` as the geometry
base, A* pathfinding on top. Grid is decoupled from the ECS world (integration later).

Subtasks:
- [x] Hex coordinate data model (honeycomb axial `{q, r}`) + hex↔world converters
      (pointy-top). → `Map/HexConfig.ts`, `HexGrid.hexToWorld`/`worldToHex`.
- [x] Grid storage structure (rectangular grid, walkability, occupancy by
      `(eid, worldId)`). → `Map/HexGrid.ts`.
- [x] Pathfinding over the grid (A* over neighbors, dynamic blocking). → `Map/findPath.ts`.
- [x] Grid rendering (instanced pointy-hex SDF overlay). → `ECS/Systems/Render/Grid/`.
- [x] Place demo tanks on random distinct cells, occupy those cells, movement
      disabled. → `createGame.spawnDemoTanks`.
- [ ] Unit movement across hexes: pick target hex → path → move the physics body
      toward the hex center (and `occupy`/`vacate` cells as it moves).
- [ ] ECS integration: wire occupancy to real game entities / separate grid world.

> When a goal is done — move it to a "Done" section and promote the next one.
