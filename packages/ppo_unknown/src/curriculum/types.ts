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
 * Leaner than tanks' 9-axis composition table: two axes — team sizes and how the
 * enemy team (team 1) behaves. Team 0 is always the learning policy.
 *
 * Enemy behaviours:
 *   standing  — no driver: a static target gallery so the policy first learns to
 *               approach, aim and fire.
 *   moving    — `RandomBot` that only occasionally steps to a random neighbour:
 *               moving targets without tactics or return fire.
 *   shooting  — `RandomBot` that wanders AND fires randomly: the learner meets
 *               incoming fire for the first time.
 *   self-play — enemies are learning agents sharing the live policy: the full
 *               competitive co-adaptation regime.
 *
 * Index ordering is load-bearing: a network's stored
 * `mapScenarioIndexToSuccessRatio` is keyed by index, so never reorder — only append.
 */

export type EnemyBehavior = 'standing' | 'moving' | 'shooting' | 'self-play';

export type ScenarioConfig = {
    /** Team 0 size — the learning policy. */
    allies: number;
    /** Team 1 size. */
    enemies: number;
    enemy: EnemyBehavior;
};

export const scenarioCompositions: readonly ScenarioConfig[] = [
    // 0: 1 vs 5 standing targets — pure approach/aim/fire practice
    { allies: 1, enemies: 1, enemy: 'standing' },
    // 1: 4 vs 4 standing targets — same skill with teammates around (friendly-fire mask matters)
    { allies: 4, enemies: 4, enemy: 'standing' },
    // 2: 4 vs 4, enemies occasionally move
    { allies: 4, enemies: 4, enemy: 'moving' },
    // 3: 4 vs 4, enemies move and shoot back
    { allies: 4, enemies: 4, enemy: 'shooting' },
    // 4: 4 vs 4 self-play
    { allies: 4, enemies: 4, enemy: 'self-play' },
] as const;

export type CurriculumState = {
    iteration: number;
    mapScenarioIndexToSuccessRatio: Record<number, number>;
};

export const DEFAULT_CURRICULUM_STATE: CurriculumState = {
    iteration: 0,
    mapScenarioIndexToSuccessRatio: {},
};
