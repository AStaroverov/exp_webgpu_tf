# Unknown — game prototype

> A clean game prototype (working name `unknown`, desert / "dune" setting) built on
> an ECS + WebGPU + Rapier foundation. This file describes how the code is organized
> and the **principles for working in the ECS** — not the game mechanics (those live
> in, and are discoverable from, the code under `src/Game`).

## Packages (monorepo)

- **`renderer`** — shared WebGPU rendering engine: GPU init, the frame loop, WGSL/SDF
  shape passes, and the base render ECS components (`Color`, `Shape`,
  `Local/GlobalTransform`, …) and systems (`Transform`, `ChangedDetector`, `Resize`).
  Game packages import it directly via relative paths (`../../../renderer/src/...`).
- **`unknown`** — *this package*. The game prototype: ECS world + Rapier physics +
  `renderer`. Runs with or without rendering (headless for training).
- **`debug`** — dev-only debug GUI overlay (`createDebugGUI`) for inspecting/tweaking
  a running game.
- **`ppo`** — standalone PPO reinforcement-learning implementation (TensorFlow.js).
- **`ppo_unknown`** — training harness that drives `unknown` headless through `ppo`
  (scenario setup + decision driver).

## Running

```bash
npm run dev      # vite dev server (config.vite.ts), HMR disabled
npm run build    # tsc + vite build
npm run preview
```

Entry point: `index.html` → `index.ts` → `src/Game/createGame.ts`. `index.ts` creates
the canvas, calls `createGame({width, height})`, `game.setRenderTarget(canvas)`, and
runs a `requestAnimationFrame` loop calling `game.gameTick(delta)` (delta clamped to
16.6667 ms). Sound is enabled lazily on a user gesture → `game.enableSound()`.

## Stack

- **ECS**: [`bitecs`](https://github.com/NateTheGreatt/biteCS) `^0.4.0`
- **Physics**: `@dimforge/rapier2d-simd` (2D, WASM)
- **Rendering**: WebGPU via `packages/renderer`
- **Reactivity**: `rxjs` + the `delegate`/`obs` component helpers
- **Hex grid**: `honeycomb-grid` (geometry, neighbors, distance) + A* on top
- **Build**: Vite 6 + `vite-plugin-wasm` + `vite-plugin-top-level-await`

## Architecture (structure only)

- **ECS world — `src/Game/ECS/createGameWorld.ts`.** Builds a `bitecs` world with a
  `{ components, time }` context. `GameComponents = RenderComponents + game components`.
  Components live in `ECS/Components/*`, created via the `defineComponent` factory and
  registered in `createGameWorld`. Read them anywhere with `getGameComponents(world)`.
- **DI singletons — `src/Game/DI/`.** Module-level objects, not classes: `GameDI`
  (world, physics, `gameTick`, public API), `RenderDI` (GPU; optional), `SoundDI`
  (lazy), `PluginDI` (`addSystem(group, …)` for `SystemGroup.{Before,After}`),
  `PlayerEnvDI`.
- **Frame — `GameDI.gameTick(delta)` (see `createGame.ts`).** Deterministic system
  order: `physicalFrame` → `plugins[Before]` → gameplay systems → camera →
  `destroyFrame` / `spawnFrame` → render/sound → `plugins[After]`. **System order is
  load-bearing** — see the ECS principles below.
- **Where things live:** `ECS/Components/*` (data), `ECS/Systems/*` (logic),
  `ECS/Entities/*` (entity factories / composition), `ECS/Actions/*` (decision-layer
  action queue, its own world), `Config/*` (re-exported via `Config/index.ts`),
  `Map/*` (hex grid, decoupled from the ECS world), `Physical/*` (Rapier glue).
  `createGame` wires only systems; build-specific world content (spawns + driver) is
  added by the caller (`setupDemoWorld` for dev, `ppo_unknown` for training).

---

# Principles for working in the ECS

The single rule everything else follows from: **components are data, systems are
behavior, and the trigger for a behavior is a query — not a conditional buried inside
a system.** When you add a feature, the question is *"what component expresses this,
and which system queries it?"* — not *"which existing system do I edit?"*.

## Anti-pattern — how NOT to add behavior

A worked example (adding "a rocket projectile explodes on impact or at the end of its
range"). The tempting-but-wrong approach was to bolt the new behavior onto whatever
systems happened to be nearby:

- **Special-casing a type inside a generic system** — `if (caliber === Rocket) { spawn
  explosion … }` inside the generic hit-processing system. The system now knows about
  one specific entity kind.
- **Adding side-effects to a single-purpose system** — making the
  "destroy-when-too-far" system *also* spawn an explosion. That system's one job is to
  add a `Destroy` marker; detonation is unrelated to it.
- **A one-off helper called from many sites** — an `explodeRocket()` function invoked
  from both the hit system and the distance system, with the *same* trigger condition
  (`is it a rocket? is it being destroyed?`) duplicated at every call site.

Why it's bad: the feature is **smeared across several systems**, each now coupled to
it; the trigger condition is **duplicated and can drift**; and every new variant
("a barrel explodes", "a mine explodes") forces yet another branch in yet another
system. Simple systems stop being simple.

## The pattern — how to add behavior

Express the behavior as **data on a component**, and let **one dedicated system** act
on the query. The same example, done right:

1. **A component carries the parameters** — `Explodable { damage, radius, vfxSize, … }`.
   It is pure data; it holds no logic.
2. **One system's query *is* the trigger** — `createExplodeSystem` queries
   `[Explodable, Destroy]` → detonate. It is ordered deliberately: it runs *before*
   the destroy system removes the entity, so the entity still exists when it explodes.
3. **The existing systems stay dumb** — they only add the `Destroy` marker. Every cause
   of destruction (collision, max range, timeout, out-of-zone) now produces the
   explosion **uniformly, with zero edits to those systems**.
4. **New explosive things just get the component** — attach `Explodable` at spawn; no
   system changes.

The trigger lives in the **combination of components** (`Explodable + Destroy`), so the
behavior is defined in exactly one place and composes with everything else.

**Generalize it:** prefer *"add a component + a system that queries it"* over
*"branch inside an existing system"* or *"call a shared helper from N sites"*. If you
find yourself writing `if (specificType)` in a generic system, that condition probably
wants to be a component. If a system does two unrelated things, split it.

## Best practices

Distilled from the sources below and our own conventions:

- **Components are data.** No behavior in components — at most trivial readonly
  helpers. Logic belongs in systems.
- **One system, one responsibility.** Depend on the *minimum* set of components: it
  reads clearer, refactors easier, and is reusable across entity kinds.
- **Don't pile unrelated fields into one component.** Split a component that mixes
  concerns into focused ones; only consolidate data that is *always* processed
  together.
- **Drive behavior with marker / "command" components, not flags + branches.** Add a
  tag (or a one-frame command component), let a system query it, and remove it after
  processing. The query selects the entities; the system never type-checks.
- **Existence-based processing — presence of a component *is* the boolean.** Don't
  store `isAlive` / `isActive` / a `state` enum and branch on it per entity. Model
  state as **membership**: a "dead" entity simply lacks the component; a transition is
  add-one-tag / remove-another, and the query does the selecting. No per-entity `if`
  in the loop body.
- **No optional / nullable fields — split them into their own component.** If a field
  is only "sometimes present" (an optional shadow caster, an optional target), make it
  a separate component instead of a nullable field you must null-check. This keeps each
  `bitecs` typed-array column dense and the query exact.
- **Relations are entity ids (numbers), not object references.** Store another entity
  as its `eid` in a component field; never hold an entity object/handle. Use a reserved
  sentinel for "none", and verify the referenced entity still exists before touching it
  (it may have been destroyed). This is also what keeps the world serializable.
- **Defer side-effects out of the query loop.** Don't fire callbacks, spawn, or destroy
  in the middle of iterating a query. Record intent as data — a marker component, or an
  append to a preallocated buffer — and act in a dedicated later system. This project
  has **no command buffer**: structural changes happen at the explicit phase boundaries
  (`destroyFrame` / `spawnFrame`), never mid-query.
- **System order is part of the design.** Sequence dependent systems so state lands in
  the same frame (avoid one-frame lag). Example: `explode()` runs before `destroy()`.
- **Keep hot loops cache-friendly and allocation-free.** Iterate query results in a
  plain `for` loop (no `forEach`, no per-element closures). Write results into reused
  scratch buffers / out-arrays; never return or build a `{x, y}` object per entity (in
  JS that heap-allocates and feeds GC spikes — the #1 cause of frame stutter). Avoid
  random cross-entity lookups inside the loop.
- **Favor refactorability over a perfect first cut.** If a behavior is genuinely
  one-off, a small helper/script is fine — promote it to a component + system as soon
  as the pattern repeats (as the explosion example did).

## Data layout & performance (JS + `bitecs`)

`bitecs` already stores components as Structure-of-Arrays (typed-array columns indexed
by `eid`), so the cache-locality win is built in — **don't fight it**. The job here is
to stay aligned with that model and with how V8 optimizes, not to hand-roll memory.

- **Keep query loops monomorphic.** One tight `for (let i = 0; i < ents.length; i++)`
  reading typed-array columns by `eid`. No `entity.update()`-style dispatch, no
  polymorphic property access, no `switch (type)` inside the kernel. If a loop needs to
  do two different things by type, that's two queries / two systems (or two components).
- **Don't reshape objects at runtime in hot paths.** Adding/deleting JS object
  properties (`delete obj.foo`) triggers hidden-class transitions and deopts. State
  changes go through component add/remove, not object mutation.
- **The fastest search is the one you don't do.** Maintain the matching set as data
  (it's what a query *is*) instead of scanning all entities and testing a condition
  every frame. For spatial proximity, build a grid / spatial hash into a **reused**
  buffer (clear-and-refill each frame) to kill O(n²) scans — never allocate it fresh.
- **Sort indices/ids, not data, and only when it pays.** A sort is justified only if it
  saves more downstream than it costs (e.g. fewer WebGPU pipeline/state switches, alpha
  ordering, early-out). Sort a `Uint32Array` of `eid`s by a precomputed key; for integer
  keys (layer, packed material id) a counting/bucket pass beats `Array.sort(cmpFn)`.
- **Double-buffer derived state** (read `current`, write `next`, swap the array refs)
  rather than rebuilding objects each frame.

### JS-specific caveats (what transfers from data-oriented design, what doesn't)

The classic DOD literature is written for C/C++. What carries over to a JS/`bitecs`
engine, and what to ignore:

- **Transfers:** SoA / typed arrays for contiguity and monomorphism; existence-based
  iteration; algorithmic wins (skip the search, spatial index, batching, deferred
  events); data-only components; sequential access over scattered random indexing;
  replacing pointer chains (`a.b.c.d`) with a single `id → index` indirection.
- **Does *not* transfer — don't attempt:** manual cache-line packing / struct padding /
  alignment, prefetch, streaming stores, aliasing hints — V8 hides memory layout.
  There is no SIMD auto-vectorization in plain JS: if you genuinely need vector kernels,
  go **WASM SIMD** or **WebGPU compute** (the SoA layout keeps data lane-ready for that).
  False sharing is irrelevant single-threaded — it only re-emerges across Web Workers
  over `SharedArrayBuffer`.
- **Inverts in JS:** "immutability everywhere" (object spreads, `Object.freeze`, a new
  object per tick) is a *pessimization* here — it creates GC pressure. Prefer in-place
  writes into typed arrays and buffer-level double-buffering.

## When NOT to optimize / abstract

- **Profile before optimizing**, after JIT warmup, and watch worst-frame percentiles
  (GC and deopt spikes hide in the average), not the mean.
- **Don't build speculative generality.** Write the concrete system first; extract a
  shared helper/abstraction only when the pattern actually repeats (3rd real use).
  Over-parameterized generic code tends to go megamorphic and deopt in V8.
- **Don't over-split components/archetypes** either: too fine adds indexing overhead,
  too coarse forces reading fields you don't use. Split when *different systems* touch
  disjoint field subsets; keep together when always co-accessed.
- **Dirty-flag laziness isn't always a win** — an unpredictable per-entity `if (dirty)`
  can be slower than just recomputing a cheap value in a straight monomorphic loop.
- Keep cold / setup code readable — abstractions, closures, and the like are fine
  *outside* hot loops.

## Sources

- [ECS — design notes (Dreaming381)](https://gist.github.com/Dreaming381/89d65f81b9b430ffead443a2d430defc)
- [Design decisions when building games using ECS (Ariel Coppes)](https://arielcoppes.dev/2023/07/13/design-decisions-when-building-games-using-ecs.html)
- [Data-Oriented Design (Richard Fabian)](https://dataorienteddesign.com/dodbook/) — the
  data-layout / performance and "when not to optimize" guidance above is distilled from
  this book, filtered for what applies to a JS/V8 + `bitecs` engine.
