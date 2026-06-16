/**
 * trainingLoop — the SINGLE-THREAD sample→learn→sample loop, the collapse of
 * ppo_unknown's actor + learner workers into one in-process driver.
 *
 * HOW THE WORKER SPLIT MAPS HERE
 * ------------------------------
 *   ppo_unknown (multi-thread)              | burn_unknown (single-thread)
 *   ----------------------------------------+--------------------------------------
 *   N ActorWorkers each run EpisodeManager  | this loop runs episodes serially in
 *   episodes headless, emit AgentMemoryBatch | the main loop, appends each finished
 *   over agentSampleChannel                 | agent's batch to an in-memory rollout
 *   ----------------------------------------+--------------------------------------
 *   LearnerManager buffers samples until    | we accumulate finished-agent batches
 *   sum(sizes) >= batchSize, runs Retrace + | until total steps >= batchSize, then
 *   PolicyLearner + ValueLearner, saves     | flatten into one batch and call
 *   weights to IndexedDB                    | V4Trainer.update (Retrace + PPO + value
 *                                           | loss + KL-LR all inside wasm)
 *   ----------------------------------------+--------------------------------------
 *   Actors reload weights from IndexedDB    | the live V4Trainer IS the latest policy
 *   each episode (sync())                   | — no reload, no version metadata
 *   ----------------------------------------+--------------------------------------
 *   metrics posted from workers over        | we post the SAME payloads on the SAME
 *   metricsChannels                         | metricsChannels (dashboard unchanged)
 *
 * Per decision tick inside an episode the driver does exactly what the real flow does:
 * snapshotUnknownBoard → computeActionMask → act → applyActionToGame → tick the game →
 * record (board, mask, action, logProb, value, reward, done) into the agent's memory.
 * Here that happens inside `BurnUnknownAgent.decide()` (driven by the reused
 * `createPolicyDriverSystem`); this module only orchestrates episodes and the update.
 */

import { metricsChannels } from "../../ppo/src/infra/channels.ts";
import { TICK_TIME_SIMULATION } from "../../ppo_unknown/src/consts.ts";
import { BOARD_SIZE } from "../../ppo_unknown/src/state/board.ts";
import { ACTION_DIM_TOTAL } from "../../ppo_unknown/src/consts.ts";
import { calculateFinalReward } from "../../ppo_unknown/src/reward/calculateReward.ts";
import type { UnknownAgent } from "../../ppo_unknown/src/env/UnknownAgent.ts";
import {
  scenarioCompositions,
  type CurriculumState,
} from "../../ppo_unknown/src/curriculum/types.ts";
import { CONFIG } from "../../ppo_unknown/src/config.ts";
import { macroTasks } from "../../../lib/TasksScheduler/macroTasks.ts";
import { type BurnMemoryBatch } from "./BurnUnknownAgent.ts";
import { createBurnScenario, type BurnScenario } from "./createBurnScenario.ts";
import { ensureTrainer, getTrainerVersion, trainerUpdate, type V4IterStats } from "./trainer.ts";

const MAX_FRAMES = CONFIG.episodeFrames;

// Wall-clock of the previous update's end (perf-clock ms), for the wait/train-time split.
let lastUpdateEnd = 0;

export type IterationReport = {
  iteration: number;
  episodes: number;
  steps: number;
  meanEpisodeReturn: number;
  meanSuccess: number;
  stats: V4IterStats;
};

/** A finished agent's batch tagged with the episode success ratio for metrics. */
type CollectedBatch = { batch: BurnMemoryBatch; successRatio: number; scenarioIndex: number };

/**
 * Run one episode of the given curriculum scenario headless and return the learning
 * agents' batches. Mirrors ppo_unknown EpisodeManager: tick until a team is wiped /
 * one tank left / the frame cap, then close + reward each agent.
 */
async function runEpisode(
  state: CurriculumState,
  shouldAbort: () => boolean,
): Promise<CollectedBatch[]> {
  // Pick the easiest viable rung for the single-thread loop: standing/moving enemies
  // are the bread-and-butter; advanced rungs (frozen/self-play) reuse the live policy.
  const index = pickScenarioIndex(state);
  const scenario = createBurnScenario({
    index,
    train: true,
    config: scenarioCompositions[index],
    iteration: state.iteration,
  });

  try {
    await runGameLoop(scenario, shouldAbort);

    const successRatio = scenario.getSuccessRatio();
    const out: CollectedBatch[] = [];
    for (const agent of scenario.agents) {
      agent.closeFinalStep();
      // calculateFinalReward only reads `.tankEid` off each agent (for team
      // grouping + contribution); BurnUnknownAgent exposes the same `tankEid`, so the
      // cast is sound — the structural-private mismatch is the only reason TS rejects it.
      const finalReward = calculateFinalReward(
        agent.tankEid,
        successRatio,
        scenario.agents as unknown as UnknownAgent[],
      );
      const batch = agent.getMemoryBatch(finalReward);
      if (batch == null) continue;
      out.push({ batch, successRatio, scenarioIndex: index });
    }

    // Report success per episode on the same channel the dashboard reads.
    metricsChannels.successRatio.postMessage([
      { scenarioIndex: index, successRatio, isReference: false },
    ]);

    return out;
  } finally {
    scenario.agents.forEach((a) => a.dispose());
    scenario.destroy();
  }
}

/**
 * Burst loop (matches AbstractEpisodeManager.runGameLoop): drain after each tick.
 * Resolves early if `shouldAbort()` flips true — lets a pressed Render toggle cut the
 * current training episode short instead of waiting out the whole match.
 */
function runGameLoop(scenario: BurnScenario, shouldAbort: () => boolean): Promise<void> {
  return new Promise((resolve) => {
    let frame = 0;
    const step = async () => {
      for (let i = 0; i < 100; i++) {
        if (shouldAbort()) return resolve();
        const gameOver = runGameTick(frame++, scenario);
        await scenario.drainDecisions();
        if (gameOver) return resolve();
      }
      macroTasks.addTimeout(step, 0); // yield so the page stays responsive
    };
    void step();
  });
}

function runGameTick(frame: number, scenario: BurnScenario): boolean {
  const aliveTanks = scenario.getVehicleEids();
  const gameOver =
    scenario.getTeamsCount() <= 1 || aliveTanks.length <= 1 || frame > MAX_FRAMES;
  scenario.gameTick(TICK_TIME_SIMULATION);
  return gameOver;
}

/** Curriculum index — promote with iteration, like the real ladder but coarse. */
function pickScenarioIndex(state: CurriculumState): number {
  // Spend the early run on the standing/moving rungs; later allow the policy rungs.
  const unlocked = state.iteration < 50 ? 2 : scenarioCompositions.length;
  return Math.floor(Math.random() * unlocked);
}

/**
 * Flatten N collected agent batches into the flat arrays V4Trainer.update expects.
 * old_logp / value are the behaviour-policy estimates captured at act() time.
 */
function flatten(batches: CollectedBatch[]): {
  boards: Float32Array;
  masks: Float32Array;
  actions: Int32Array;
  oldLogp: Float32Array;
  rewards: Float32Array;
  dones: Float32Array;
  values: Float32Array;
  steps: number;
} {
  let steps = 0;
  for (const { batch } of batches) steps += batch.size;

  const boards = new Float32Array(steps * BOARD_SIZE);
  const masks = new Float32Array(steps * ACTION_DIM_TOTAL);
  const actions = new Int32Array(steps);
  const oldLogp = new Float32Array(steps);
  const rewards = new Float32Array(steps);
  const dones = new Float32Array(steps);
  const values = new Float32Array(steps);

  let s = 0;
  for (const { batch } of batches) {
    for (let i = 0; i < batch.size; i++, s++) {
      boards.set(batch.states[i].board, s * BOARD_SIZE);
      if (batch.masks) masks.set(batch.masks[i], s * ACTION_DIM_TOTAL);
      actions[s] = batch.actions[i][0] | 0;
      oldLogp[s] = batch.logProbs[i];
      rewards[s] = batch.rewards[i];
      dones[s] = batch.dones[i];
      values[s] = batch.values[i];
    }
  }

  return { boards, masks, actions, oldLogp, rewards, dones, values, steps };
}

/**
 * Run ONE training iteration: collect the FIRST episode that yields any samples, run a
 * single V4Trainer.update on it, then publish metrics. Per the user, we no longer wait to
 * accumulate ~batchSize steps across many episodes — we update as soon as the first batch
 * arrives. This keeps iterations short (~one episode) so logs/metrics flow continuously
 * and the Render toggle is picked up quickly (index.ts only re-checks it between iters).
 * We still loop past episodes that collected 0 steps (e.g. every agent died before opening
 * a step), since an empty batch is nothing to update on.
 */
export async function runTrainingIteration(
  state: CurriculumState,
  shouldAbort: () => boolean = () => false,
): Promise<IterationReport | null> {
  await ensureTrainer();

  const collected: CollectedBatch[] = [];
  let steps = 0;
  let episodes = 0;
  let returnSum = 0;
  let returnCount = 0;
  let successSum = 0;

  while (steps === 0) {
    if (shouldAbort()) return null; // Render requested → bail without an update.
    const episodeBatches = await runEpisode(state, shouldAbort);
    episodes += 1;
    for (const cb of episodeBatches) {
      collected.push(cb);
      steps += cb.batch.size;
      // Mean episodic return = sum of recorded step rewards (incl. terminal).
      let epReturn = 0;
      for (let i = 0; i < cb.batch.rewards.length; i++) epReturn += cb.batch.rewards[i];
      returnSum += epReturn;
      returnCount += 1;
      successSum += cb.successRatio;
    }
    console.log(
      `[collect] iter=${state.iteration} ep#${episodes} +${
        episodeBatches.reduce((a, b) => a + b.batch.size, 0)
      } → ${steps} steps`,
    );
  }

  // Abort fired while the (productive) episode was finishing → discard the truncated
  // batch and let the caller switch to Render rather than train on a half-match.
  if (shouldAbort()) return null;

  const flat = flatten(collected);

  // Timing, mirroring the tfjs learner (createLearnerManager): waitTime = the gap since
  // the previous update finished (here that gap IS the episode-collection time, since the
  // single thread can't sample and train at once), trainTime = the update itself. Both in
  // seconds, posted on the same channels the dashboard charts.
  const trainStart = performance.now();
  const waitTime = lastUpdateEnd === 0 ? undefined : (trainStart - lastUpdateEnd) / 1000;

  const stats = await trainerUpdate(
    flat.boards,
    flat.masks,
    flat.actions,
    flat.oldLogp,
    flat.rewards,
    flat.dones,
    flat.values,
  );

  lastUpdateEnd = performance.now();
  const trainTime = (lastUpdateEnd - trainStart) / 1000;

  publishMetrics(stats, flat, collected, trainTime, waitTime);

  return {
    iteration: getTrainerVersion(),
    episodes,
    steps: flat.steps,
    meanEpisodeReturn: returnCount > 0 ? returnSum / returnCount : 0,
    meanSuccess: returnCount > 0 ? successSum / returnCount : 0,
    stats,
  };
}

/**
 * Publish per-iteration metrics on the EXACT channels + payload shapes the dashboard
 * (MetricsBrowser) subscribes to, so pressing M shows live charts unchanged.
 *   - kl/policyLoss/valueLoss/entropy/lr → number[] (store.add(...list))
 *   - rewards/values → Float32Array
 *   - batchSize/versionDelta → number[]
 * V4Trainer reports only the FINAL-epoch scalars (kl/policy/value/entropy), so each
 * is published as a single-element list — the charts treat them as one point.
 */
function publishMetrics(
  stats: V4IterStats,
  flat: ReturnType<typeof flatten>,
  collected: CollectedBatch[],
  trainTime: number,
  waitTime: number | undefined,
) {
  metricsChannels.kl.postMessage([stats.kl]);
  metricsChannels.lr.postMessage([stats.lr]);
  metricsChannels.policyLoss.postMessage([stats.policy_loss]);
  metricsChannels.valueLoss.postMessage([stats.value_loss]);
  metricsChannels.entropy.postMessage([stats.entropy]);

  metricsChannels.rewards.postMessage(flat.rewards);
  metricsChannels.values.postMessage(flat.values);
  metricsChannels.batchSize.postMessage(collected.map((c) => c.batch.size));

  // Seconds; waitTime is undefined on the very first iteration (no prior update to gap from).
  metricsChannels.trainTime.postMessage([trainTime]);
  if (waitTime !== undefined) metricsChannels.waitTime.postMessage([waitTime]);
}
