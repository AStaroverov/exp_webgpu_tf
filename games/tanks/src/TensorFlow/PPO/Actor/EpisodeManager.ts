import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { TankAgent } from '../../Common/Curriculum/Agents/ActorAgent.ts';
import { CONFIG } from '../config.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { CurriculumState, curriculumStateChannel, episodeSampleChannel } from '../channels.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { createScenarioByCurriculumState } from '../../Common/Curriculum/createScenarioByCurriculumState.ts';

export class EpisodeManager {
    protected curriculumState: CurriculumState = {
        mapScenarioIndexToSuccessRatio: {},
    };

    constructor() {
        curriculumStateChannel.obs.subscribe((state) => {
            this.curriculumState = state;
        });
    }

    public async start() {
        while (true) {
            try {
                await this.runEpisode();
            } catch (error) {
                console.error('Error during episode:', error);
            }
        }
    }

    protected async beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {
            withRender: false,
            withPlayer: false,
        });
    }

    protected afterEpisode(episode: Scenario) {
        episode.getAgents().forEach(agent => {
            if (agent.getVersion == null || agent.getMemoryBatch == null) return;
            episodeSampleChannel.emit({
                networkVersion: agent.getVersion(),
                memoryBatch: agent.getMemoryBatch(),
                successRatio: episode.getSuccessRatio(),
                scenarioIndex: episode.index,
            });
        });
    }

    protected cleanupEpisode(episode: Scenario) {
        episode.destroy();
    }

    protected async runEpisode() {
        const episode = await this.beforeEpisode();

        try {
            await this.runGameLoop(episode);
            this.afterEpisode(episode);
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(episode);
        }
    }

    protected runGameLoop(episode: Scenario) {
        return new Promise(resolve => {
            const shouldEvery = SNAPSHOT_EVERY;
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            let regardedActors: TankAgent[] = [];
            let frame = 0;

            const stop = macroTasks.addInterval(() => {
                for (let i = 0; i < 100; i++) {
                    frame++;
                    const nextRegardedActors = this.runGameTick(
                        episode,
                        regardedActors,
                        frame,
                        maxFramesCount,
                        shouldEvery,
                    );

                    if (nextRegardedActors == null) {
                        stop();
                        resolve(null);
                        break;
                    } else {
                        regardedActors = nextRegardedActors;
                    }
                }
            }, 1);
        });
    }

    protected runGameTick(
        scenario: Scenario,
        prevActors: TankAgent[],
        frame: number,
        maxFrames: number,
        shouldEvery: number,
    ) {
        const actors = scenario.getActors();
        const currentTanks = scenario.getTankEids();
        const gameOverByActorCount = actors.length <= 0;
        const gameOverByTankCount = currentTanks.length <= 1;
        const gameOverByTeamWin = scenario.getTeamsCount() === 1;
        const gameOverByTime = frame > maxFrames;
        const gameOver = gameOverByActorCount || gameOverByTankCount || gameOverByTeamWin || gameOverByTime;
        const shouldAction = frame % shouldEvery === 0;
        const shouldMemorize = gameOver || (frame - (shouldEvery - 1)) % shouldEvery === 0;
        TenserFlowDI.shouldCollectState = (frame + 1) % shouldEvery === 0;

        if (shouldAction) {
            prevActors = actors;

            for (const agent of prevActors) {
                agent.updateTankBehaviour(GameDI.width, GameDI.height);
            }
        }

        // Execute game tick
        scenario.gameTick(TICK_TIME_SIMULATION);

        if (shouldMemorize) {
            for (const agent of prevActors) {
                agent.memorizeTankBehaviour?.(
                    GameDI.width,
                    GameDI.height,
                    gameOver,
                );
            }
        }

        return gameOver ? null : prevActors;
    }
}
