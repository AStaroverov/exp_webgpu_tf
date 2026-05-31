# Plan: `ppo_unknown` — PPO training for the hex-grid game

> Design decisions locked in:
> - **Action space:** high-level discrete actions tied to the chess-like action queue (decide when the queue empties).
> - **Reward / win-lose logic:** lives entirely inside `ppo_unknown` (the `unknown` game package stays untouched).
> - This document is the implementation plan; no code is written yet.

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
- `ppo_unknown` uses **high-level discrete actions** keyed to *"when the action queue empties"*. Each decision is therefore a **macro-action (option)** spanning a *variable* number of game ticks: a `MoveToHex` may run dozens of ticks; a `Wait` only a few.

Implications the implementation must respect:

- A **"step"** in PPO terms = **one decision point**, not one game tick.
- The reward attached to a transition is the **sum of per-tick rewards accumulated over the entire macro-action duration**. v1: undiscounted sum within the option, γ applied between options. If unstable, fall back to discounting inside the option or capping macro-action length.
- `EpisodeManager.runGameTick` still runs every physics tick (to advance Rapier + the action systems), but it invokes the agent's *decide → record* logic **only on ticks where that agent's action queue is empty / its action finished**.
- `AgentMemory<S>` two-phase API fits this exactly, no changes to `packages/ppo`:
  - `addFirstPart(state, action, logits, logProb)` → called **at the decision point**.
  - `updateSecondPart(reward, done)` → called **at the next decision point** with the accumulated reward.

---

## 3. Where the agent plugs into the game

The `unknown` game exposes everything we need (confirmed during exploration):

- **Action queue API:** `enqueueAction(eid, { kind, target, params })` for `ActionKind.{MoveToHex, TurretAim, Fire, Wait}`; the action world reports status (`Idle` / `Running` / `Finished`).
  - `MoveToHex`: `target: { kind: TargetKind.Hex, q, r }`, `params: { speed }`
  - `TurretAim`: `target: { kind: TargetKind.Entity, eid }`, `params: { tolerance }`
  - `Fire`: `params: { shots }`
  - `Wait`: timer-based
- **State readers:** `RigidBodyState.position / rotation / linearVelocity / angularVelocity`, `getTankHealth(eid)`, `getTankCurrentPartsCount(eid)`, `TeamRef.id[eid]`, `Firearms.isReloading(eid)`, bullet queries `query(world, [Bullet])`.
- **Hex map:** `MapDI.grid` → `worldToHex(x,y)`, `neighbors(q,r)`, `isPassable(q,r)`, `getOccupant(q,r)`, and `findPath(grid, start, goal)`.
- **Headless:** `createGame()` without `setRenderTarget` runs physics + actions + gameplay with **no GPU** — ideal for actor workers.

**Decision:** the game package stays untouched. The agent lives entirely in `ppo_unknown` and drives the game purely through `enqueueAction` + the public state readers. We add a small helper to detect *"this agent's queue is empty / action finished"* by querying the action world for actions owned by the agent's `eid`.

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
    │   ├── UnknownAgent.ts          # decide loop, fills AgentMemory<S>
    │   ├── agentQueue.ts            # "is this agent's queue empty?" helpers vs action world
    │   └── applyActionToGame.ts     # sampled action indices -> enqueueAction calls
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

Categorical multi-head policy:

| Head | Dim | Meaning |
|------|-----|---------|
| `[0]` kind | 4 | `{ MoveToHex, TurretAim, Fire, Wait }` |
| `[1]` moveTargetHex | `K_HEX` | index into a fixed, agent-relative enumeration of candidate hex cells (e.g. all hexes within radius 2 ⇒ 19) |
| `[2]` aimTarget | `K_ENEMY` | index into a fixed-size enemy slot list (e.g. 3 nearest enemies), masked |
| `[3]` fireShots | 3 | `{ 1, 2, 3 }` |

`ACTION_HEAD_DIMS = [4, K_HEX, K_ENEMY, 3]`.

`applyActionToGame(agent, actionIndices)`:
- read `kind` from head `[0]`;
- issue **exactly one** `enqueueAction` for that kind, using the relevant head(s); heads irrelevant to the chosen kind are ignored;
- map `moveTargetHex` index → concrete `{q, r}` via the agent-relative enumeration;
- map `aimTarget` index → concrete enemy `eid` via the enemy slot list.

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

- `beforeEpisode(): Scen` → `createUnknownScenario()`: headless `createGame`, spawn N-vs-M tanks on the hex grid, wrap each learning tank in an `UnknownAgent`; return the scenario (game handle, agents, team info).
- `runGameTick(frame, dt, scen): boolean`:
  1. `scen.game.gameTick(dt)`;
  2. for each agent, accumulate this tick's per-tick reward into its running tally;
  3. if the agent's action queue is empty → `agent.decide(state)`:
     - first close the *previous* step via `updateSecondPart(accumulatedReward, done)`,
     - then `addFirstPart(state, action, logits, logProb)`,
     - then `applyActionToGame` and reset the reward tally;
  4. return `isEpisodeDone(scen)`.
- `awaitAgentsSync()`: pull the latest network version (same mechanism as tanks `CurrentActorAgent`).
- `afterEpisode(scen)`: finalize each agent's memory with the final reward, emit `agentSampleChannel` per agent + `episodeSampleChannel` with the success ratio.
- `cleanupEpisode(scen)`: `scen.game.destroy()`.

### 5.6 Config & consts — `config.ts`, `consts.ts`

Copy `ppo_tanks` `PpoConfig` as the starting point: clip `0.2`, `4` epochs each, adaptive entropy, KL-driven LR, batch `4096` / mini-batch `256`. Then tune:

- `episodeFrames` — decision-based episodes are short in *steps* but long in *ticks*; set a tick cap of a few game-minutes.
- `workerCount` — start 2–4.
- `savePath: 'PPO_UNKNOWN'`.
- `consts.ts`: `ACTION_HEAD_DIMS`, `TICK_TIME_SIMULATION`, `LEARNING_STEPS`, the `K_*` slot sizes.

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

- **Action masking** with the generic `batchAct` (samples raw logits): v1 = no-op fallback + penalty; v2 = inject `-inf` masked logits. Decide where to apply without modifying `packages/ppo`.
- **Variable option length** can make returns noisy; if unstable, discount *within* the option or cap macro-action duration.
- **Hex candidate enumeration** for the move head must be a stable, fixed-size, agent-relative ordering, or the policy can't learn a consistent mapping.
- **No win condition in the game** means `ppo_unknown` owns termination — ensure bullets / destroyed entities are cleaned between episodes to avoid leaks across the long-lived worker.
- **Action world scope:** confirm `ActionScheduleDI`'s global stack is per-game-instance, not a process-wide singleton. Current plan runs one game per actor at a time, so it is safe — but verify before any future multi-game-per-worker change.
