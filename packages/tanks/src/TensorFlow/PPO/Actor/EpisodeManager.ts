import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from 'rxjs';
import { abs, max, min } from '../../../../../../lib/math.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { createScenarioByCurriculumState } from '../../Common/Curriculum/createScenarioByCurriculumState.ts';
import { Scenario } from '../../Common/Curriculum/types.ts';
import { CurriculumState, curriculumStateChannel, episodeSampleChannel, queueSizeChannel } from '../channels.ts';
import { CONFIG } from '../config.ts';
import { GAME_OVER_REWARD_MULTIPLIER } from '../../Reward/calculateReward.ts';

const queueSize$ = queueSizeChannel.obs.pipe(
    startWith(0),
    shareReplay(1),
);
const backpressure$ = race([
    timer(60_000),
    queueSize$.pipe(filter((queueSize) => queueSize <= CONFIG.backpressureQueueSize)),
]).pipe(first());

const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % SNAPSHOT_EVERY) + SNAPSHOT_EVERY);

export class EpisodeManager {
    protected curriculumState: CurriculumState = {
        currentVersion: 0,
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

    protected beforeEpisode() {
        return createScenarioByCurriculumState(this.curriculumState, {
            iteration: this.curriculumState.currentVersion,
        });
    }

    protected afterEpisode(episode: Scenario) {
        episode.getPilots().forEach(agent => {
            if (agent.getVersion == null || agent.getMemoryBatch == null) {
                return;
            }

            const successRatio = episode.getSuccessRatio();
            const memoryBatch = agent.getMemoryBatch(successRatio * GAME_OVER_REWARD_MULTIPLIER);

            if (memoryBatch == null) {
                return;
            }

            const minReward = min(...memoryBatch.rewards);
            const maxReward = max(...memoryBatch.rewards);

            if (abs(minReward) < 1 && abs(maxReward) < 1) {
                // Skip if the rewards are too small, indicating no significant learning
                console.info('Skipping episode sample due to low reward magnitude:', { minReward, maxReward });
                return;
            }

            episodeSampleChannel.emit({
                networkVersion: agent.getVersion(),
                memoryBatch: memoryBatch,
                successRatio,
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
            let frame = 0;

            const stop = macroTasks.addInterval(() => {
                try {
                    for (let i = 0; i < 100; i++) {
                        const gameOver = this.runGameTick(
                            frame++,
                            TICK_TIME_SIMULATION,
                            episode,
                        );

                        if (gameOver) {
                            stop();
                            resolve(null);
                            break;
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
        frame: number,
        deltaTime: number,
        scenario: Scenario,
    ) {
        const actors = scenario.getAliveActors();
        const currentTanks = scenario.getTankEids();
        const gameOverByActorCount = actors.length <= 0;
        const gameOverByTankCount = currentTanks.length <= 1;
        const gameOverByTeamWin = scenario.getTeamsCount() === 1;
        const gameOverByTime = frame > maxFramesCount;
        const gameOver = gameOverByActorCount || gameOverByTankCount || gameOverByTeamWin || gameOverByTime;

        scenario.gameTick(deltaTime);

        return gameOver;
    }
}
