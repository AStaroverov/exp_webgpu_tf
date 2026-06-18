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
 * scenarioCompositions — the ppo_unknown scenario set.
 *
 * TRAINING follows a learnable-frontier autocurriculum ladder (see
 * `createScenarioByCurriculumState`): scenarios unlock by an annealing success
 * threshold and are softmax-weighted toward the harder / less-mastered ones. The
 * population diversity the self-play literature wants falls out of this — mastered
 * scripted bots fade to the ε-floor while self-play and frozen past-self (never
 * saturated) stay near the frontier — so no separate opponent-mix table is needed.
 * The scenarios ALSO double as EVAL BASELINES, sampled on reference (greedy, non-train)
 * episodes to measure how the live policy fares against each fixed opponent. Their
 * per-index rolling success ratio is both the "how good is the bot right now" gauge in
 * the metrics UI AND the signal the ladder reads; only reference episodes update it.
 *
 * Storage index order below is load-bearing (success ratios are keyed by it, so never
 * reorder — only append); the ladder walks an explicit difficulty order instead.
 *
 * Two axes — team sizes and how the enemy team (team 1) behaves. Team 0 is always the
 * learning policy.
 *
 * Enemy behaviours:
 *   standing  — `RandomBot` that holds position but occasionally fires down a
 *               random direction: a near-static target gallery with sporadic return fire.
 *   moving    — `RandomBot` that occasionally steps to a random neighbour AND
 *               occasionally fires: moving targets with undirected return fire.
 *   bot-hunter — `HunterBot`: a deterministic A*-chaser that aims and closes. With a
 *               `hunterReactionMs` reaction delay it recomputes its plan only every N ms
 *               of simulated time and re-commits the stale one in between — a slower,
 *               more exploitable hunter (a lag-handicapped difficulty knob).
 *   frozen    — enemies run a FROZEN historical snapshot of the policy
 *               (`FrozenAgent`): a stable opponent — measures progress vs past selves.
 *   self-play — enemies are learning agents sharing the live policy: the full
 *               competitive co-adaptation regime. THE training scenario.
 *
 * Index ordering is load-bearing: a network's stored
 * `mapScenarioIndexToSuccessRatio` is keyed by index, so never reorder — only append.
 */

export type EnemyBehavior = "bot-standing" | "bot-moving" | "bot-hunter" | "frozen" | "self-play";

export type ScenarioConfig = {
  maxCount: number;
  enemy: EnemyBehavior;
  /** Only for `bot-hunter`: HunterBot reaction lag in simulated ms (0 = instant). */
  hunterReactionMs?: number;
};

export const scenarioCompositions: readonly ScenarioConfig[] = [
  { maxCount: 4, enemy: "self-play" },
  { maxCount: 4, enemy: "bot-standing" },
  { maxCount: 4, enemy: "bot-moving" },
  { maxCount: 4, enemy: "bot-hunter", hunterReactionMs: 1000 },
  { maxCount: 4, enemy: "bot-hunter", hunterReactionMs: 2000 },
  { maxCount: 4, enemy: "bot-hunter" },
  { maxCount: 4, enemy: "frozen" },
] as const;

export type CurriculumState = {
  iteration: number;
  mapScenarioIndexToSuccessRatio: Record<number, number>;
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
  iteration: 0,
  mapScenarioIndexToSuccessRatio: {},
};
