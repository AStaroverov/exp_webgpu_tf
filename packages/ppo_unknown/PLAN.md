# Plan: `ppo_unknown` — PPO training for the hex-grid game

> Design decisions locked in:
> - **Action space:** high-level discrete actions tied to the per-owner action queue (decide whenever the owner presents an open slot — `needsDecision(eid)`).
> - **Reward / win-lose logic:** lives entirely inside `ppo_unknown` (the `unknown` game package stays untouched).
> - This document is the implementation plan; no code is written yet.
>
> **Updated 2026-06-02** to reflect the `unknown` action-system refactor (commit `948308f`):
> the action queue now lives ON the entity (`ActionsQueue` component, `MAX_QUEUE = 2`),
> there is no separate action ECS world / `ActionScheduleDI`, action kinds are
> `{ MoveStep, Aim, Fire, Hold }` (atomic — `MoveStep` is a single hop), and the
> decision seam is a `PluginDI` `SystemGroup.Before` system (the placeholder
> `createStandInDriverSystem` is what the ML policy will replace).

---

## 1. Guiding principle

The generic `packages/ppo` package is **fully game-agnostic** and is reused verbatim:

- Training losses — `trainPolicyNetwork`, `trainValueNetwork`
- Off-policy targets — `computeRetraceTargets` (Retrace(λ))
- KL / LR adaptation — `computeKullbackLeiblerAprox`, `getDynamicLearningRate`
- Learner orchestration — `createLearnerManager`, `createPolicyLearnerAgent`, `createValueLearnerAgent`
- Memory — `AgentMemory<S>` / `AgentMemoryBatch<S>`
- Channels — `agentSampleChannel`, `episodeSampleChannel`, `learnProcessChannel`, `queueSizeChannel`, `modelSettingsChannel`, metrics channels
- Hyperparameter type — `PpoConfig`

We implement **only** the game-specific seams that `packages/ppo` declares:

1. `StateBindings<S>` — encode game state → input tensors.
2. `EpisodeManager<Scen>` — the episode loop.
3. An agent that fills `AgentMemory<S>`.
4. Policy / value network factories (categorical logit heads + scalar value).
5. `actionHeadDims`, a `PpoConfig` instance, the reward function, and the worker entry points.

`packages/ppo_tanks` is the structural template — copy its layout and replace the game-coupled parts.

---

## 2. Central design challenge: the semi-MDP / options model

This is the one place `ppo_unknown` genuinely diverges from `ppo_tanks`, and the whole plan is built around it.

- `ppo_tanks` decides on a **fixed cadence** (every ~6 ticks via `SNAPSHOT_EVERY`).
- `ppo_unknown` uses **high-level discrete actions** keyed to *"when the owner presents an open slot"* (`needsDecision(eid)`). Each decision is therefore a **macro-action (option)** spanning a *variable* number of game ticks: a `MoveStep` (one hop to a neighbour) runs until the body reaches the cell; a `Hold` only a few ticks.
- The atomic-action refactor makes options **shorter and more uniform** than the original `MoveToHex`-to-arbitrary-hex plan assumed — `MoveStep` is a single hex hop, so a multi-hex route is now several decisions, not one. Each executor raises its `requestNext` flag *near* completion (`MoveStep` near the destination, `Aim`/`Fire` immediately, `Hold` near timer end), opening the slot so the next decision is pre-decided one step ahead (`MAX_QUEUE = 2` = one running + one queued).

Implications the implementation must respect:

- A **"step"** in PPO terms = **one decision point** (`needsDecision` hit), not one game tick.
- The reward attached to a transition is the **sum of per-tick rewards accumulated over the entire macro-action duration**. v1: undiscounted sum within the option, γ applied between options. If unstable, fall back to discounting inside the option or capping macro-action length. (Shorter atomic options make this less of a risk than the original plan assumed.)
- The agent's *decide → record* logic runs inside a `SystemGroup.Before` plugin (the seam where `createStandInDriverSystem` sits today), invoked **only for owners where `needsDecision(eid)` is true** on that tick. The physics + action executor + scheduler systems still run every tick to advance the simulation.
- `AgentMemory<S>` two-phase API fits this exactly, no changes to `packages/ppo`:
  - `addFirstPart(state, action, logits, logProb)` → called **at the decision point**.
  - `updateSecondPart(reward, done)` → called **at the next decision point** with the accumulated reward.

---

## 3. Where the agent plugs into the game

The `unknown` game exposes everything we need (confirmed against the post-refactor code):

- **Action queue API** (`ECS/Actions/ActionSchedule.ts` — thin free functions over the game world):
  - `needsDecision(eid): boolean` — the decision seam: queue has room (`count < MAX_QUEUE`) **and** the owner presents an open slot (idle, or the running front raised `requestNext`).
  - `enqueueAction(eid, spec): boolean` — encode one atomic action into the next free slot; returns `false` if the queue is full.
  - `queueDepth(eid)`, `isIdle(eid)` — read helpers. `MAX_QUEUE = 2`.
  - Actions live ON the entity as the `ActionsQueue` component (slot 0 = front); there is **no separate action world and no `ActionScheduleDI`** anymore — the SoA buffers are encapsulated, callers only touch the free functions above.
  - `EnqueueActionSpec` is derived from the per-kind descriptor registry. `ActionKind.{ MoveStep, Aim, Fire, Hold }`, `TargetKind.{ None, Entity, Hex, Point }`, `ActionStatus.{ Idle, Running, Finished }`.
    - `MoveStep`: `target: { kind: TargetKind.Hex, q, r }` (a **single neighbour hop** — reserves the target cell `free → Reserved → Unit`), `params: { speed }`.
    - `Aim`: `target: Hex`, `params: { tolerance }` (rotate the turret toward the hex).
    - `Fire`: `target: Hex` — **self-contained**: aims the turret at the hex, waits for reload, fires **one round** (no `shots` param anymore).
    - `Hold`: `params: { duration }` (tactical timer pause; replaces `Wait`).
- **State readers:** `RigidBodyState.position / rotation / linearVelocity / angularVelocity`, `getTankHealth(eid)` / `getTankCurrentPartsCount(eid)` (`Entities/Tank/TankUtils.ts`), `TeamRef.id[eid]`, `Firearms.isReloading(eid)`, bullet queries `query(world, [Bullet])`.
- **Hex map:** `MapDI.grid` → `worldToHex(x,y)`, `neighbors({q,r})`, `isPassable(q,r)`, `occupy/vacate/getOccupant` (`OccupantKind.{ ..., Reserved }`), plus `findPath` in `Map/findPath.ts`. Dynamic occupancy is maintained by `createGridOccupancySystem`.
- **Headless:** `createGame()` without `setRenderTarget` runs physics + actions + gameplay with **no GPU** — ideal for actor workers.

**Decision:** the game package stays untouched. The agent lives entirely in `ppo_unknown` and drives the game through the documented decision seam — exactly the seam `createStandInDriverSystem` uses today. The ML policy is a `PluginDI` `SystemGroup.Before` system that, per tick, iterates living tanks, skips those where `!needsDecision(eid)`, and for the rest runs *decide → record → `enqueueAction`*. No custom queue-introspection helper is needed (`needsDecision` / `isIdle` / `queueDepth` already exist).

---

## 4. Package structure (mirrors `ppo_tanks`)

```
packages/ppo_unknown/
├── package.json            # copy ppo_tanks deps; add dependency on the `unknown` game pkg
├── tsconfig.json
├── config.vite.ts
├── index.ts
└── src/
    ├── consts.ts           # action head config, TICK_TIME, decision cadence, LEARNING_STEPS
    ├── config.ts           # PpoConfig instance (start from ppo_tanks values)
    ├── entry/
    │   ├── index.ts                 # spawn actors + 2 learners + optional visualizer
    │   ├── ActorWorker.ts           # WASM backend, runs UnknownEpisodeManager
    │   ├── LearnerPolicyWorker.ts   # WebGPU backend, policy learner
    │   └── LearnerValueWorker.ts    # WebGPU backend, value learner
    ├── env/                # game-integration layer (the genuinely new code)
    │   ├── createUnknownScenario.ts # headless createGame, spawn teams on hex grid, return Scen
    │   ├── createPolicyDriverSystem.ts # SystemGroup.Before plugin: iterate needsDecision tanks → decide → record (replaces createStandInDriverSystem)
    │   ├── UnknownAgent.ts          # per-tank decide loop, fills AgentMemory<S>
    │   └── applyActionToGame.ts     # sampled action indices -> enqueueAction calls
    │                                # NOTE: no agentQueue.ts — needsDecision/isIdle/queueDepth already exist in ActionSchedule.ts
    ├── state/
    │   ├── bindings.ts              # StateBindings<InputArrays> implementation
    │   ├── InputArrays.ts           # game state -> typed arrays (the S type)
    │   └── InputTensors.ts          # batch S[] -> tf.Tensor[]
    ├── models/
    │   ├── dims.ts                  # input dims + ACTION_HEAD_DIMS
    │   ├── Inputs.ts                # tf input layers + tokenization
    │   ├── createUnknownNetworks.ts # policy + value factories
    │   └── Networks/v1.ts           # start from a simplified tanks v13
    ├── reward/
    │   └── calculateReward.ts       # per-tick reward + episode-end outcome
    ├── agents/
    │   └── UnknownEpisodeManager.ts # extends EpisodeManager<Scen>
    └── ui/                          # optional: port tanks visualizer / metrics later
```

---

## 5. Module-by-module plan

### 5.1 Action space — `models/dims.ts`, `env/applyActionToGame.ts`

Categorical multi-head policy. The atomic actions changed the natural head layout:
`MoveStep` is a **single neighbour hop** (≤ 6 directions, `POINTY_DIRECTIONS`), and
`Fire` now **self-aims at a hex** and fires one round (no shot-count, no separate aim
needed for the common case).

| Head | Dim | Meaning |
|------|-----|---------|
| `[0]` kind | 4 | `{ MoveStep, Aim, Fire, Hold }` |
| `[1]` moveDir | 6 | index into `POINTY_DIRECTIONS` (the agent's 6 neighbour hexes); resolves to `{q, r}` of that neighbour |
| `[2]` fireTarget | `K_ENEMY` | index into a fixed-size enemy slot list (e.g. 3 nearest enemies), masked → that enemy's hex for `Fire`/`Aim` |

`ACTION_HEAD_DIMS = [4, 6, K_ENEMY]`.

`applyActionToGame(agent, actionIndices)`:
- read `kind` from head `[0]`;
- issue **exactly one** `enqueueAction` for that kind, using the relevant head(s); heads irrelevant to the chosen kind are ignored;
- `MoveStep`: map `moveDir` index → the neighbour `{q, r}` via `grid.neighbors(here)` in a stable `POINTY_DIRECTIONS` order;
- `Fire` / `Aim`: map `fireTarget` index → the chosen enemy's current hex (`worldToHex` of its body) via the enemy slot list;
- `Hold`: a fixed `duration` from consts.

> Open design choice: whether the agent needs an explicit `Aim` head at all, since
> `Fire` already self-aims. v1 can fold `Aim` into `Fire` and keep the kind head at
> 3 (`{ MoveStep, Fire, Hold }`) — revisit if turret pre-aiming proves useful.

**Invalid-action handling** (aim with no enemy, move to an occupied/impassable hex, fire while reloading):
- v1 (cheapest): no-op fallback in `applyActionToGame` + small invalid-action penalty in the reward.
- v2 (cleaner): build an `actionMask(S)` and inject `-inf` into masked logits before sampling. Confirm where to apply it without touching `packages/ppo` (`batchAct` samples raw logits — masking must happen in the network output or in a wrapper).

### 5.2 Observation encoding — `state/`

Define `S = InputArrays` as plain typed arrays (like tanks). Candidate fields:

- **Self:** hp, hex `(q, r)` normalized, world pos, `cos/sin` rotation, velocity, `isReloading` flag, `queueBusy` flag.
- **Turret:** `cos/sin` of turret angle, reload progress.
- **Enemies** (up to `K_ENEMY`): relative hex offset, relative world pos, hp, relative velocity, validity mask.
- **Allies** (up to `K_ALLY`): same shape + mask.
- **Local hex patch:** occupancy / passability grid around the agent (e.g. radius-2 patch) with channels `{ passable, ally, enemy, self }`.
- **Bullets** (optional, up to `K_BULLET`): relative pos / vel + mask.

`InputTensors.ts` batches `S[]` → `tf.Tensor[]`. `bindings.ts` implements `createInputTensors(batch)` + `prepareRandomInputArrays()`. Reuse the `norm` / `logNorm` helpers from tanks.

### 5.3 Networks — `models/`

Start from a **simplified** tanks `v13` (full Perceiver is overkill for v1):
- tokenize each group (self, enemies, allies, hex patch);
- concat tokens;
- a couple of dense / light attention layers;
- **Policy:** 4 categorical heads sized by `ACTION_HEAD_DIMS`.
- **Value:** single scalar.

Both are consumed unchanged by `createPolicyLearnerAgent` / `createValueLearnerAgent`. Keep `Inputs.ts` parametrized by `dims.ts` so observation changes don't ripple through the model code.

### 5.4 Reward & termination — `reward/calculateReward.ts`

Computed from game state **inside `ppo_unknown`**. Per-tick shaping accumulated across the macro-action:

- `+` damage dealt (inspect `Hitable` hits / enemy hp deltas), `+` kill bonus.
- `−` damage taken / parts lost, `−` friendly fire.
- small `+` for closing distance to the nearest enemy.
- small `−` per-decision time cost (discourage stalling).
- `−` invalid / no-op action penalty.

Episode-end outcome (team-spirit style, reuse the tanks formula): `+win` / `−loss` scaled by team share.

**Termination** — `isEpisodeDone(scen)`:
- one team fully destroyed (alive query over `TeamRef`), **or**
- `episodeFrames` tick cap reached.

Also expose `getSuccessRatio(scen)` for the `episodeSampleChannel` feedback (metrics / future curriculum).

### 5.5 Episode manager — `agents/UnknownEpisodeManager.ts`

`extends EpisodeManager<Scen>`:

- `beforeEpisode(): Scen` → `createUnknownScenario()`: headless `createGame`, spawn N-vs-M tanks on the hex grid, wrap each learning tank in an `UnknownAgent`, and install `createPolicyDriverSystem(agents)` as a `SystemGroup.Before` plugin **in place of** `createStandInDriverSystem` (don't run both). Return the scenario (game handle, agents, team info).
- `runGameTick(frame, dt, scen): boolean`:
  1. for each agent, accumulate this tick's per-tick reward into its running tally (before the tick so deltas are measured against the pre-tick state, or after — pick one and keep it consistent);
  2. `scen.game.gameTick(dt)` — this runs the `SystemGroup.Before` policy-driver plugin, which for every tank with `needsDecision(eid) === true` calls `agent.decide(state)`:
     - first close the *previous* step via `updateSecondPart(accumulatedReward, done)`,
     - then `addFirstPart(state, action, logits, logProb)`,
     - then `applyActionToGame` (→ `enqueueAction`) and reset the reward tally;
  3. return `isEpisodeDone(scen)`.

> The decide → record logic physically lives inside the `SystemGroup.Before` plugin
> (so it sees the same `needsDecision` gate the game defines), but it is owned and
> wired by `ppo_unknown`. `runGameTick` is just the per-tick reward accumulator + the
> `gameTick` pump + termination check.
- `awaitAgentsSync()`: pull the latest network version (same mechanism as tanks `CurrentActorAgent`).
- `afterEpisode(scen)`: finalize each agent's memory with the final reward, emit `agentSampleChannel` per agent + `episodeSampleChannel` with the success ratio.
- `cleanupEpisode(scen)`: `scen.game.destroy()`.

### 5.6 Config & consts — `config.ts`, `consts.ts`

Copy `ppo_tanks` `PpoConfig` as the starting point: clip `0.2`, `4` epochs each, adaptive entropy, KL-driven LR, batch `4096` / mini-batch `256`. Then tune:

- `episodeFrames` — decision-based episodes are short in *steps* but long in *ticks*; set a tick cap of a few game-minutes.
- `workerCount` — start 2–4.
- `savePath: 'PPO_UNKNOWN'`.
- `consts.ts`: `ACTION_HEAD_DIMS = [4, 6, K_ENEMY]`, `TICK_TIME_SIMULATION`, `LEARNING_STEPS`, the enemy/ally/bullet slot sizes (`K_ENEMY`, `K_ALLY`, `K_BULLET`). The move head is a fixed 6 (`POINTY_DIRECTIONS`), so there is no `K_HEX` for actions; a separate radius for the observation hex patch (§5.2) is its own const.

### 5.7 Entry / workers — `entry/`

Copy the 4-file worker layout from `ppo_tanks` verbatim, swapping in `UnknownEpisodeManager`, `createUnknownNetworks`, and `ACTION_HEAD_DIMS`:

- Actors → WASM backend.
- Learners (policy + value) → WebGPU backend.
- Main tab → spawns actors + learners + optional visualizer.

---

## 6. Phasing (milestones)

1. **Scaffold + headless smoke test** — `package.json` / `tsconfig` / `vite`; `createUnknownScenario` spawns a headless game, ticks N frames, prints alive counts. No RL yet.
2. **Agent loop, no learning** — `UnknownAgent` with a random / scripted policy via `enqueueAction`; verify queue-empty detection and per-macro-action reward accumulation.
3. **State + networks** — `InputArrays` / `InputTensors` / `bindings`, `createUnknownNetworks`; verify a forward pass yields correctly-shaped logits / value on a real observation.
4. **Wire learners** — `actionHeadDims`, `createLearnerManager` + policy / value agents in workers; confirm batches flow `actor → agentSampleChannel → learner` and weights update.
5. **Reward shaping + termination** — implement `calculateReward` + `isEpisodeDone`; first end-to-end training run; watch the metrics channels (reward, KL, entropy, value loss).
6. **Tune + visualize** — port the metrics / visualizer UI, tune hyperparameters, optionally add a curriculum (`scenarioCompositions`) and self-play later.

---

## 7. Open questions / risks to validate during build

- **Action masking** with the generic `batchAct` (samples raw logits): v1 = no-op fallback + penalty; v2 = inject `-inf` masked logits. Decide where to apply without modifying `packages/ppo`. (Masking is now cheaper to reason about: `moveDir` legality = `isPassable` of each of the 6 neighbours; `fireTarget` legality = enemy-slot validity.)
- **Variable option length** can make returns noisy; if unstable, discount *within* the option or cap macro-action duration. (Lower risk now that `MoveStep` is a single, short hop rather than a full path.)
- **Move head ordering** must follow a stable `POINTY_DIRECTIONS` index so neighbour slot N always means the same direction; missing/impassable neighbours are masked or no-op'd, not reindexed.
- **No win condition in the game** means `ppo_unknown` owns termination — ensure bullets / destroyed entities are cleaned between episodes to avoid leaks across the long-lived worker.
- **Single decision driver:** the policy plugin must *replace* `createStandInDriverSystem`, not run alongside it (two drivers would both fill slots). Confirm `createUnknownScenario` either omits the stand-in or removes it before adding the policy driver.
- ~~**Action world scope:**~~ **RESOLVED by the refactor** — there is no separate action world or `ActionScheduleDI` global stack anymore. The queue is the per-entity `ActionsQueue` component, so it is inherently per-game-instance; multi-game-per-worker is safe on this axis.
