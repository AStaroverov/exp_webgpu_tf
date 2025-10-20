import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from 'rxjs';
import { abs, max, min } from '../../../../../lib/math.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { SAC_CONFIG } from '../../../../ml-common/config.ts';
import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../../../../ml-common/consts.ts';
import { createScenarioByCurriculumState } from '../../../../ml-common/Curriculum/createScenarioByCurriculumState.ts';
import { Scenario } from '../../../../ml-common/Curriculum/types.ts';
import { getGameOverReward } from '../../Reward/calculateReward.ts';
import { CurriculumState, curriculumStateChannel, episodeSampleChannel, queueSizeChannel } from '../channels.ts';

const queueSize$ = queueSizeChannel.obs.pipe(
    startWith(0),
    shareReplay(1),
);
const backpressure$ = race([
    timer(60_000),
    queueSize$.pipe(filter((queueSize) => queueSize <= SAC_CONFIG.backpressureQueueSize)),
]).pipe(first());

const maxFramesCount = (SAC_CONFIG.episodeFrames - (SAC_CONFIG.episodeFrames % SNAPSHOT_EVERY) + SNAPSHOT_EVERY);

/**
 * SAC Episode Manager
 * Collects experience in the form of (s, a, r, s', done) transitions
 * No need to compute advantages or returns - that's done in the training loop
 */
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
            // Используем getMemoryBatch и конвертируем в SAC формат
            // nextStates создаются автоматически из states[i+1]
            if (agent.getVersion == null || agent.getMemoryBatch == null) {
                return;
            }

            const networkVersion = agent.getVersion();
            const successRatio = episode.getSuccessRatio();
            const finalReward = getGameOverReward(networkVersion, successRatio);
            const agentBatch = agent.getMemoryBatch(finalReward);

            if (agentBatch == null || agentBatch.size <= 1) {
                // Нужно минимум 2 элемента, чтобы создать nextStates
                return;
            }

            // Конвертируем AgentMemoryBatch в SACMemoryBatch
            // Обрезаем на 1, создавая nextStates из states[i+1]
            const size = agentBatch.size - 1;
            const memoryBatch = {
                size,
                states: agentBatch.states.slice(0, size),
                actions: agentBatch.actions.slice(0, size),
                rewards: agentBatch.rewards.slice(0, size),
                nextStates: agentBatch.states.slice(1), // states[1..n]
                dones: agentBatch.dones.slice(0, size),
                perturbed: agentBatch.perturbed.slice(0, size),
            };

            const minReward = min(...memoryBatch.rewards);
            const maxReward = max(...memoryBatch.rewards);

            if (abs(minReward) < 1 && abs(maxReward) < 1) {
                // Skip if the rewards are too small, indicating no significant learning
                console.info('Skipping episode sample due to low reward magnitude:', { minReward, maxReward });
                return;
            }

            episodeSampleChannel.emit({
                memoryBatch: memoryBatch,
                successRatio,
                scenarioIndex: episode.index,
                networkVersion,
            });
        });
    }

    protected cleanupEpisode(episode: Scenario) {
        episode.destroy();
    }

    protected async runEpisode() {
        const episode = await this.beforeEpisode();

        try {
            await this.awaitAgentsSync(episode);
            await this.runGameLoop(episode);
            this.afterEpisode(episode);
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(episode);
        }
    }

    protected awaitAgentsSync(episode: Scenario) {
        return Promise.all(episode.getPilots().map(agent => agent.sync?.()));
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
