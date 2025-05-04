import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { TankAgent } from '../../Common/Curriculum/Agents/CurrentActorAgent.ts';
import { CONFIG } from '../config.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { CurriculumState, curriculumStateChannel, episodeSampleChannel, queueSizeChannel } from '../channels.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { createScenarioByCurriculumState } from '../../Common/Curriculum/createScenarioByCurriculumState.ts';
import { snapshotTankInputTensor } from '../../../ECS/Utils/snapshotTankInputTensor.ts';
import { abs, max, min } from '../../../../../../lib/math.ts';
import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from 'rxjs';

const queueSize$ = queueSizeChannel.obs.pipe(
    startWith(0),
    shareReplay(1),
);
const backpressure$ = race([
    timer(60_000),
    queueSize$.pipe(filter((queueSize) => queueSize < 3)),
]).pipe(first());

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
                await firstValueFrom(backpressure$);
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
            if (agent.getVersion == null || agent.getMemoryBatch == null) {
                return;
            }

            const memoryBatch = agent.getMemoryBatch();
            const minReward = min(...memoryBatch.rewards);
            const maxReward = max(...memoryBatch.rewards);

            if (abs(minReward) < 2 && abs(maxReward) < 2) {
                // Skip if the rewards are too small, indicating no significant learning
                return;
            }

            episodeSampleChannel.emit({
                networkVersion: agent.getVersion(),
                memoryBatch,
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
        return new Promise((resolve, reject) => {
            const shouldEvery = SNAPSHOT_EVERY;
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            let regardedAgents: TankAgent[] = [];
            let frame = 0;

            const stop = macroTasks.addInterval(() => {
                try {
                    for (let i = 0; i < 100; i++) {
                        frame++;
                        const nextRegardedAgents = this.runGameTick(
                            TICK_TIME_SIMULATION,
                            episode,
                            regardedAgents,
                            frame,
                            maxFramesCount,
                            shouldEvery,
                        );

                        if (nextRegardedAgents == null) {
                            stop();
                            resolve(null);
                            break;
                        } else {
                            regardedAgents = nextRegardedAgents;
                        }
                    }
                } catch (error) {
                    stop();
                    reject(error);
                }
            }, 1);
        });
    }

    protected runGameTick(
        deltaTime: number,
        scenario: Scenario,
        currentAgents: TankAgent[],
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

        if (shouldAction || gameOver) {
            for (const agent of currentAgents) {
                agent.evaluateTankBehaviour?.(GameDI.width, GameDI.height, gameOver);
            }
        }

        if (gameOver) {
            return null;
        }

        if (shouldAction) {
            snapshotTankInputTensor();

            currentAgents = scenario.getAgents();

            for (const agent of currentAgents) {
                agent.updateTankBehaviour(GameDI.width, GameDI.height);
            }
        }

        scenario.gameTick(deltaTime);

        return currentAgents;
    }
}
