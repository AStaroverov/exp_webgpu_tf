/**
 * ppo_unknown CONFIG — a `PpoConfig` instance consumed unchanged by the generic
 * learner/actor in `packages/ppo`. Started from `ppo_tanks/src/config.ts`; only
 * `savePath` and episode sizing diverge for the hex game.
 */

import type { PpoConfig } from "../../ppo/src/config.ts";
import { TICK_TIME_SIMULATION } from "./consts.ts";

export const CONFIG: PpoConfig & {
  episodeFrames: number;
  workerCount: number;
} = {
  clipNorm: 5,

  // Effective horizon 1/(1-γ): 0.97 ≈ 33 decisions — too short for a 60–240-decision
  // episode, so the terminal win/loss reward never reaches the early/mid trajectory.
  // 0.99 ≈ 100 decisions lets the sparse terminal signal carry the bulk of the episode
  // (required once the dense shaping is annealed away). Tunable: 0.98 if returns get noisy.
  gamma: (_iteration: number) => 0.98,

  entropyCoeff: 0.03,

  policyEpochs: (_iter: number) => 10,
  policyClipRatio: 0.2,
  policyLogitsL2: 1e-3,

  valueEpochs: (_iter: number) => 10,
  valueClipRatio: 0.2,
  valueLossCoeff: 0.5,
  valueLRCoeff: 1,

  lrConfig: {
    kl: { high: 0.013, target: 0.01, low: 0.007 },
    initial: 1e-4,
    multHigh: 0.95,
    multLow: 1.05,
    min: 1e-5,
    max: 1e-3,
  },

  batchSize: (_iteration: number) => 2048,
  miniBatchSize: (_iteration: number) => 512,

  // Decision-based episodes are short in STEPS but long in TICKS. Cap ticks at
  // a few simulated minutes; termination is owned by ppo_unknown (no win cond
  // baked into the game).
  episodeFrames: Math.round((2 * 60 * 1000) / TICK_TIME_SIMULATION),
  workerCount: 5,
  backpressureQueueSize: 2,
  savePath: "PPO_UNKNOWN_V3",
};
