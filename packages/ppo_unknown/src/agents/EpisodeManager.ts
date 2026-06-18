/**
 * UnknownEpisodeManager — the ppo_unknown episode loop. Subclasses the generic
 * `EpisodeManager<Scenario>` exactly like tanks' `TankEpisodeManager`: each episode
 * is sampled from the curriculum ladder (see `scenarioCompositions`) by
 * `createScenarioByCurriculumState`, driven by the curriculum state the learner
 * broadcasts over `curriculumStateChannel`.
 *
 * Per episode: build a headless scenario, pull fresh policy weights, tick until a
 * team is wiped / one tank left / the frame cap, then emit each learning tank's
 * `AgentMemoryBatch` (with its terminal reward) and the episode success ratio.
 */

import { max } from "../../../../lib/math.ts";
import { random } from "../../../../lib/random.ts";
import { AbstractEpisodeManager } from "../../../ppo/src/core/EpisodeManager.ts";
import { agentSampleChannel, episodeSampleChannel } from "../../../ppo/src/core/channels.ts";
import { CONFIG } from "../config.ts";
import { TICK_TIME_SIMULATION } from "../consts.ts";
import { calculateFinalReward } from "../reward/calculateReward.ts";
import { Scenario } from "../env/createUnknownScenario.ts";
import { UnknownAgent } from "../env/UnknownAgent.ts";
import { FrozenAgent } from "../env/FrozenAgent.ts";
import { createScenarioByCurriculumState } from "../curriculum/createScenarioByCurriculumState.ts";
import { CurriculumState, DEFAULT_CURRICULUM_STATE } from "../curriculum/types.ts";
import { curriculumStateChannel } from "../curriculumChannel.ts";

const MAX_FRAMES = CONFIG.episodeFrames;

/**
 * Share (0..1) of the single most frequent value in `values`.
 *
 * Used to flag a degenerate ("stuck") episode where one reward repeats. Since
 * the caller's threshold is above 0.5, the most frequent value is by definition
 * the majority element, so Boyer–Moore finds the candidate in O(1) space (no
 * Map / float-key hashing) and a second pass counts its true share.
 */
function majorityShare(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;

  let candidate = 0;
  let votes = 0;
  for (let i = 0; i < values.length; i++) {
    if (votes === 0) candidate = values[i];
    votes += values[i] === candidate ? 1 : -1;
  }

  let count = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] === candidate) count++;
  }
  return count / values.length;
}

export class EpisodeManager extends AbstractEpisodeManager<Scenario> {
  protected curriculumState: CurriculumState = DEFAULT_CURRICULUM_STATE;

  constructor() {
    super({
      backpressureQueueSize: CONFIG.backpressureQueueSize,
      simulationTickTime: TICK_TIME_SIMULATION,
    });
    curriculumStateChannel.obs.subscribe((state) => {
      this.curriculumState = state;
    });
  }

  protected beforeEpisode(): Scenario {
    return createScenarioByCurriculumState(this.curriculumState, { train: random() < 0.95 });
  }

  protected afterEpisode(scenario: Scenario): void {
    const successRatio = scenario.getSuccessRatio();
    const isReference = !scenario.train;

    episodeSampleChannel.emit({
      maxNetworkVersion: max(...scenario.agents.map((a) => a.getVersion()), 0),
      scenarioIndex: scenario.index,
      successRatio,
      isReference,
    });

    if (isReference) return;

    scenario.agents.forEach((agent) => {
      agent.closeFinalStep();
      const networkVersion = agent.getVersion();
      const finalReward = calculateFinalReward(agent.tankEid, successRatio, scenario.agents);
      const memoryBatch = agent.getMemoryBatch(finalReward);
      if (memoryBatch == null) return;

      // Detect a "stuck" episode: when the same reward value repeats for almost
      // the whole batch the simulation likely froze.
      const sameRewardsProcent = majorityShare(memoryBatch.rewards);
      if (sameRewardsProcent > 0.9) {
        console.warn(
          `Skipping sample with identical rewards ${Math.round(sameRewardsProcent * 100)}% (scenario=${scenario.index}, version=${networkVersion}, size=${memoryBatch.size})`,
        );
        return;
      }

      agentSampleChannel.emit({
        networkVersion,
        scenarioIndex: scenario.index,
        memoryBatch,
      });
    });
  }

  protected cleanupEpisode(scenario: Scenario): void {
    scenario.agents.forEach((a) => a.dispose());
    scenario.destroy();
  }

  protected drainDecisions(scenario: Scenario): Promise<void> {
    return scenario.drainDecisions();
  }

  protected awaitAgentsSync(): Promise<unknown> {
    // FrozenAgent.sync() also re-rolls which historical version the frozen
    // opponents play this episode.
    return Promise.all([UnknownAgent.sync(), FrozenAgent.sync()]);
  }

  protected runGameTick(frame: number, deltaTime: number, scenario: Scenario): boolean {
    const aliveTanks = scenario.getVehicleEids();
    const gameOverByTeamWin = scenario.getTeamsCount() <= 1;
    const gameOverByTankCount = aliveTanks.length <= 1;
    const gameOverByTime = frame > MAX_FRAMES;
    const gameOver = gameOverByTeamWin || gameOverByTankCount || gameOverByTime;

    scenario.gameTick(deltaTime);

    return gameOver;
  }
}
