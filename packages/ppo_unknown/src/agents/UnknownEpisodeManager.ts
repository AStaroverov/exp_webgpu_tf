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

import { max } from '../../../../lib/math.ts';
import { random } from '../../../../lib/random.ts';
import { EpisodeManager } from '../../../ppo/src/core/EpisodeManager.ts';
import { agentSampleChannel, episodeSampleChannel } from '../../../ppo/src/core/channels.ts';
import { CONFIG } from '../config.ts';
import { TICK_TIME_SIMULATION } from '../consts.ts';
import { calculateFinalReward } from '../reward/calculateReward.ts';
import { Scenario } from '../env/createUnknownScenario.ts';
import { UnknownAgent } from '../env/UnknownAgent.ts';
import { FrozenAgent } from '../env/FrozenAgent.ts';
import { createScenarioByCurriculumState } from '../curriculum/createScenarioByCurriculumState.ts';
import { CurriculumState, DEFAULT_CURRICULUM_STATE } from '../curriculum/types.ts';
import { curriculumStateChannel } from '../curriculumChannel.ts';

const MAX_FRAMES = CONFIG.episodeFrames;

export class UnknownEpisodeManager extends EpisodeManager<Scenario> {
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
        return createScenarioByCurriculumState(this.curriculumState, { train: random() < 0.9 });
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
            const finalReward = calculateFinalReward(agent.tankEid, scenario.getTeamDestroyedRatio, scenario.agents);
            const memoryBatch = agent.getMemoryBatch(finalReward);
            if (memoryBatch == null) return;

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
