/**
 * createScenarioByCurriculumState — pick the next ppo_unknown training scenario
 * from the curriculum ladder, then build its world.
 *
 * Same sampling policy as tanks
 * (`packages/ppo_tanks/src/curriculum/createScenarioByCurriculumState.ts`):
 *   - an unlock threshold that anneals upward with the network iteration;
 *   - scenarios unlock one ahead of the highest one the agent has cleared;
 *   - among unlocked scenarios, softmax-weight toward the harder/less-mastered ones,
 *     boost any that have regressed, and keep an ε-floor so none starves.
 *
 * The only ppo_unknown specifics: the index maps to a `ScenarioConfig`
 * (`scenarioCompositions`), and the world is built by `createUnknownScenario`.
 */

import { createUnknownScenario, Scenario } from '../env/createUnknownScenario.ts';
import { CurriculumState, scenarioCompositions } from './types.ts';

export const scenariosCount = scenarioCompositions.length;
export function createScenarioByCurriculumState(
    curriculumState: CurriculumState,
    options: { train?: boolean },
): Scenario {
    const threshold = getCurrentThreshold(curriculumState.iteration);
    const unlockedCount = getUnlockedCount(curriculumState, threshold);
    const weights = computeSamplingWeights(curriculumState, unlockedCount, threshold);
    const index = sampleIndex(weights);

    return createUnknownScenario({
        index,
        train: options.train,
        config: scenarioCompositions[index],
        iteration: curriculumState.iteration,
    });
}

const THRESHOLD_STEP_ITERATIONS = 500;
const THRESHOLD_MIN = 0.55;
const THRESHOLD_MAX = 0.75;
const THRESHOLD_STEP = 0.02;

const SOFTMAX_BETA = 3;
const EPSILON_FLOOR = 0.05;
const REGRESSION_DROP = 0.1;
const REGRESSION_BOOST = 3;

function getCurrentThreshold(iteration: number): number {
    return Math.min(THRESHOLD_MIN + THRESHOLD_STEP * Math.floor(iteration / THRESHOLD_STEP_ITERATIONS), THRESHOLD_MAX);
}

function getSuccessRatio(state: CurriculumState, index: number): number {
    return state.mapScenarioIndexToSuccessRatio[index] ?? 0;
}

// Unlocked set spans [0, highestPassed + 1]. Scan all indices so a regression
// on an earlier scenario does not re-lock later ones the agent has already cleared.
function getUnlockedCount(state: CurriculumState, threshold: number): number {
    let highestPassed = -1;
    for (let i = 0; i < scenariosCount; i++) {
        if (getSuccessRatio(state, i) >= threshold) {
            highestPassed = i;
        }
    }
    const unlocked = Math.min(highestPassed + 2, scenariosCount);
    return Math.max(unlocked, 1);
}

function sampleIndex(weights: number[]): number {
    const total = weights.reduce((s, w) => s + w, 0);
    const r = Math.random() * total;
    let acc = 0;
    for (let i = 0; i < weights.length; i++) {
        acc += weights[i];
        if (r < acc) return i;
    }
    return weights.length - 1;
}

function computeSamplingWeights(state: CurriculumState, unlockedCount: number, threshold: number): number[] {
    const rawWeights: number[] = [];
    for (let i = 0; i < unlockedCount; i++) {
        const raw = getSuccessRatio(state, i);
        const normalized = (raw + 1) / 2;
        let w = Math.exp(SOFTMAX_BETA * (1 - normalized));
        // Regression: a previously competent scenario has degraded — prioritize recovery.
        if (raw < threshold - REGRESSION_DROP) {
            w *= REGRESSION_BOOST;
        }
        rawWeights.push(w);
    }

    const sum = rawWeights.reduce((s, w) => s + w, 0);
    const probs = rawWeights.map(w => w / sum);

    // ε-floor guarantees every unlocked scenario keeps non-trivial replay probability.
    const floored = probs.map(p => Math.max(p, EPSILON_FLOOR));
    const flooredSum = floored.reduce((s, p) => s + p, 0);
    return floored.map(p => p / flooredSum);
}
