/**
 * createScenarioByCurriculumState — pick the next ppo_unknown scenario and build it.
 *
 * TRAINING (`train: true`) is a learnable-frontier autocurriculum (the regime that was
 * here before self-play existed, restored): an unlock threshold that anneals upward
 * with the network iteration; scenarios unlock one ahead of the hardest one the agent
 * has cleared; among unlocked ones, softmax-weight toward the harder/less-mastered,
 * boost any that have regressed, and keep an ε-floor so none starves. This subsumes a
 * fixed opponent mix: mastered scripted bots fade to the floor while self-play / frozen
 * past-self (never saturated) stay near the top — i.e. the population diversity the
 * self-play literature wants falls out of the frontier weighting.
 *
 * Storage index order in `scenarioCompositions` is load-bearing (success ratios are
 * keyed by it) so the ladder can't reorder it; it instead walks the scenarios in an
 * explicit difficulty order (`LADDER_ORDER`, easiest → hardest, self-play last) and
 * maps back to the storage index.
 *
 * REFERENCE episodes (`train: false`, greedy, no memory) instead probe a random EVAL
 * BASELINE (every non-self-play scenario). Their per-index rolling success ratio (kept
 * in the learner, shown in the metrics UI) is the gauge the ladder reads; only
 * reference episodes update it, so it reflects exploitation skill, not noisy exploration.
 *
 * The index maps to a `ScenarioConfig` (`scenarioCompositions`); the world is built by
 * `createUnknownScenario`.
 */

import { random } from "../../../../lib/random.ts";
import { createUnknownScenario, Scenario } from "../env/createUnknownScenario.ts";
import { CurriculumState, EnemyBehavior, scenarioCompositions } from "./types.ts";

export const scenariosCount = scenarioCompositions.length;

// ── Frontier ladder tuning (restored from the pre-self-play curriculum) ──────────
const THRESHOLD_STEP_ITERATIONS = 500;
const THRESHOLD_MIN = 0.55;
const THRESHOLD_MAX = 0.75;
const THRESHOLD_STEP = 0.02;

const SOFTMAX_BETA = 3;
const EPSILON_FLOOR = 0.05;
const REGRESSION_DROP = 0.1;
const REGRESSION_BOOST = 3;

/**
 * Per-behaviour difficulty rank (easiest → hardest). The ladder walks storage indices
 * sorted by this, so self-play (hardest) is the terminal rung even though it sits at
 * storage index 0. Within `bot-hunter`, a longer reaction lag is easier, so laggier
 * variants come first.
 */
const DIFFICULTY_RANK: Record<EnemyBehavior, number> = {
  "bot-standing": 0,
  "bot-moving": 1,
  "bot-hunter": 2,
  frozen: 3,
  "self-play": 4,
};

// Storage indices in ascending difficulty; ladder positions index into this.
const LADDER_ORDER: number[] = scenarioCompositions
  .map((_, i) => i)
  .sort((a, b) => {
    const ca = scenarioCompositions[a];
    const cb = scenarioCompositions[b];
    const d = DIFFICULTY_RANK[ca.enemy] - DIFFICULTY_RANK[cb.enemy];
    if (d !== 0) return d;
    // Same behaviour (the hunter variants): more reaction lag = easier = earlier.
    return (cb.hunterReactionMs ?? 0) - (ca.hunterReactionMs ?? 0);
  });

export function createScenarioByCurriculumState(
  curriculumState: CurriculumState,
  options: { train?: boolean },
): Scenario {
  const index = pickFrontierIndex(curriculumState);

  return createUnknownScenario({
    index,
    train: options.train,
    config: scenarioCompositions[index],
    iteration: curriculumState.iteration,
  });
}

/** Sample a storage index from the unlocked, frontier-weighted ladder. */
function pickFrontierIndex(state: CurriculumState): number {
  const threshold = getCurrentThreshold(state.iteration);
  const unlockedCount = getUnlockedCount(state, threshold);
  const weights = computeSamplingWeights(state, unlockedCount, threshold);
  return LADDER_ORDER[sampleLadderPosition(weights)];
}

function getCurrentThreshold(iteration: number): number {
  return Math.min(
    THRESHOLD_MIN + THRESHOLD_STEP * Math.floor(iteration / THRESHOLD_STEP_ITERATIONS),
    THRESHOLD_MAX,
  );
}

function getSuccessRatio(state: CurriculumState, storageIndex: number): number {
  return state.mapScenarioIndexToSuccessRatio[storageIndex] ?? 0;
}

// Unlocked set spans ladder positions [0, highestPassed + 1]. Scan all positions so a
// regression on an earlier rung does not re-lock later ones already cleared.
function getUnlockedCount(state: CurriculumState, threshold: number): number {
  let highestPassed = -1;
  for (let pos = 0; pos < LADDER_ORDER.length; pos++) {
    if (getSuccessRatio(state, LADDER_ORDER[pos]) >= threshold) {
      highestPassed = pos;
    }
  }
  const unlocked = Math.min(highestPassed + 2, LADDER_ORDER.length);
  return Math.max(unlocked, 1);
}

function sampleLadderPosition(weights: number[]): number {
  const total = weights.reduce((s, w) => s + w, 0);
  const r = random() * total;
  let acc = 0;
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i];
    if (r < acc) return i;
  }
  return weights.length - 1;
}

function computeSamplingWeights(
  state: CurriculumState,
  unlockedCount: number,
  threshold: number,
): number[] {
  const rawWeights: number[] = [];
  for (let pos = 0; pos < unlockedCount; pos++) {
    const raw = getSuccessRatio(state, LADDER_ORDER[pos]);
    const normalized = (raw + 1) / 2;
    let w = Math.exp(SOFTMAX_BETA * (1 - normalized));
    // Regression: a previously competent rung has degraded — prioritize recovery.
    if (raw < threshold - REGRESSION_DROP) {
      w *= REGRESSION_BOOST;
    }
    rawWeights.push(w);
  }

  const sum = rawWeights.reduce((s, w) => s + w, 0);
  const probs = rawWeights.map((w) => w / sum);

  // ε-floor guarantees every unlocked rung keeps non-trivial replay probability.
  const floored = probs.map((p) => Math.max(p, EPSILON_FLOOR));
  const flooredSum = floored.reduce((s, p) => s + p, 0);
  return floored.map((p) => p / flooredSum);
}
