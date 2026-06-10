/**
 * Curriculum state for ppo_unknown — the same shape tanks uses
 * (`packages/ppo_tanks/src/curriculum/types.ts`): a rolling per-scenario success
 * ratio plus the network iteration the thresholds anneal against. Persisted in the
 * policy network's user-defined metadata (see `curriculumMeta.ts`) and broadcast to
 * the actors over `curriculumStateChannel`.
 *
 * `Scenario` itself lives with the world factory (`env/createUnknownScenario.ts`);
 * re-exported here so curriculum consumers import a single module.
 */

export type { Scenario } from "../env/createUnknownScenario.ts";

/**
 * scenarioCompositions — the ppo_unknown curriculum ladder.
 *
 * Leaner than tanks' 9-axis composition table: two axes — team sizes and how the
 * enemy team (team 1) behaves. Team 0 is always the learning policy.
 *
 * Enemy behaviours:
 *   standing  — `RandomBot` that holds position but occasionally fires down a
 *               random direction: a near-static target gallery with sporadic
 *               return fire, so the policy first learns to approach, aim and fire.
 *   moving    — `RandomBot` that occasionally steps to a random neighbour AND
 *               occasionally fires: moving targets with undirected return fire.
 *   frozen    — enemies run a FROZEN historical snapshot of the policy
 *               (`FrozenAgent`): a stable opponent before live co-adaptation.
 *   self-play — enemies are learning agents sharing the live policy: the full
 *               competitive co-adaptation regime.
 *
 * Index ordering is load-bearing: a network's stored
 * `mapScenarioIndexToSuccessRatio` is keyed by index, so never reorder — only append.
 * (The 'shooting' rung and its index were dropped here when standing/moving gained
 * sporadic return fire of their own; ratios are re-keyed from scratch since the v3
 * observation change already forces a retrain.)
 */

export type EnemyBehavior = "standing" | "moving" | "frozen" | "self-play";

export type ScenarioConfig = {
  /** Team 0 size — the learning policy. */
  allies: number;
  /** Team 1 size. */
  enemies: number;
  enemy: EnemyBehavior;
};

export const scenarioCompositions: readonly ScenarioConfig[] = [
  // 1: 4 vs 4 near-static targets — same skill with a full team (friendly-fire mask matters)
  { allies: 4, enemies: 4, enemy: "standing" },
  // 2: 4 vs 4, enemies occasionally move and fire
  { allies: 4, enemies: 4, enemy: "moving" },
  // 3: 4 vs 4 against a frozen historical snapshot of the policy
  { allies: 4, enemies: 4, enemy: "frozen" },
  // 4: 4 vs 4 self-play
  { allies: 4, enemies: 4, enemy: "self-play" },
] as const;

export type CurriculumState = {
  iteration: number;
  mapScenarioIndexToSuccessRatio: Record<number, number>;
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
  iteration: 0,
  mapScenarioIndexToSuccessRatio: {},
};
