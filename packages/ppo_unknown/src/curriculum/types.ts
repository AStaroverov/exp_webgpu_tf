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

export type { Scenario } from '../env/createUnknownScenario.ts';

/**
 * scenarioCompositions — the ppo_unknown curriculum ladder.
 *
 * Far leaner than tanks' 9-axis composition table: ppo_unknown keeps a fixed
 * N-vs-N self-play world and varies a SINGLE axis — how the enemy team (team 1)
 * is driven. Team 0 is always the learning policy.
 *
 *   0 standing   — enemies hold position (no driver): a static target gallery so
 *                  the policy first learns to approach, aim and fire.
 *   1 random     — enemies wander with a light scripted bot (`RandomBot`): moving
 *                  targets, but no tactics to exploit the learner.
 *   2 self-play  — enemies are learning agents sharing the live policy (the v1
 *                  behaviour): the full competitive co-adaptation regime.
 *
 * Index ordering is load-bearing: a network's stored
 * `mapScenarioIndexToSuccessRatio` is keyed by index, so never reorder — only append.
 */

export type EnemyKind = 'standing' | 'random' | 'self-play';

export const scenarioCompositions: readonly EnemyKind[] = ['standing', 'random', 'self-play'] as const;

export type CurriculumState = {
    iteration: number;
    mapScenarioIndexToSuccessRatio: Record<number, number>;
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
    iteration: 0,
    mapScenarioIndexToSuccessRatio: {},
};
