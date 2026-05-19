import { max } from '../../../../lib/math.ts';
import { random } from '../../../../lib/random.ts';
import { EpisodeManager } from '../../../ppo/src/core/EpisodeManager.ts';
import { agentSampleChannel, episodeSampleChannel } from '../../../ppo/src/core/channels.ts';
import { CONFIG } from '../config.ts';
import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../consts.ts';
import { createScenarioByCurriculumState } from '../curriculum/createScenarioByCurriculumState.ts';
import { CurriculumState, Scenario } from '../curriculum/types.ts';
import { curriculumStateChannel } from '../curriculumChannel.ts';
import { getAliveLearnableAgents, getRegistratedAgents, Pilot } from '../../../tanks/src/Plugins/Pilots/Components/Pilot.ts';
import { calculateFinalReward } from '../reward/calculateReward.ts';

const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % SNAPSHOT_EVERY) + SNAPSHOT_EVERY);

export class TankEpisodeManager extends EpisodeManager<Scenario> {
    protected curriculumState: CurriculumState = {
        iteration: 0,
        mapScenarioIndexToSuccessRatio: {},
    };

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
        return createScenarioByCurriculumState(this.curriculumState, {
            train: random() < 0.9,
            iteration: this.curriculumState.iteration,
        });
    }

    protected afterEpisode(episode: Scenario) {
        const successRatio = episode.getSuccessRatio();
        const isReference = !episode.train;
        const pilots = Array.from(Pilot.agent.values());

        episodeSampleChannel.emit({
            maxNetworkVersion: max(...pilots.map(p => p.getVersion?.() ?? 0)),
            scenarioIndex: episode.index,
            successRatio,
            isReference,
        });

        if (isReference) return;

        pilots.forEach(agent => {
            if (agent.getVersion == null || agent.getMemoryBatch == null) {
                return;
            }

            const networkVersion = agent.getVersion();
            const finalReward = calculateFinalReward(agent.tankEid, successRatio, pilots, episode.index);
            const memoryBatch = agent.getMemoryBatch(finalReward);

            if (memoryBatch == null) return;

            agentSampleChannel.emit({
                networkVersion,
                scenarioIndex: episode.index,
                memoryBatch,
            });
        });
    }

    protected cleanupEpisode(episode: Scenario) {
        episode.destroy();
    }

    protected awaitAgentsSync() {
        return Promise.all(getRegistratedAgents().map(agent => agent.sync?.()));
    }

    protected runGameTick(
        frame: number,
        deltaTime: number,
        scenario: Scenario,
    ): boolean {
        const actors = getAliveLearnableAgents();
        const currentTanks = scenario.getVehicleEids();
        const gameOverByActorCount = actors.length <= 0;
        const gameOverByTankCount = currentTanks.length <= 1;
        const gameOverByTeamWin = scenario.getTeamsCount() === 1;
        const gameOverByTime = frame > maxFramesCount;
        const gameOver = gameOverByActorCount || gameOverByTankCount || gameOverByTeamWin || gameOverByTime;

        scenario.gameTick(deltaTime);

        return gameOver;
    }
}
