import { filter, first, firstValueFrom, race, shareReplay, startWith, timer } from 'rxjs';
import { max } from '../../../../../lib/math.ts';
import { random } from '../../../../../lib/random.ts';
import { macroTasks } from '../../../../../lib/TasksScheduler/macroTasks.ts';
import { CONFIG } from '../../../../ml-common/config.ts';
import { SNAPSHOT_EVERY, TICK_TIME_SIMULATION } from '../../../../ml-common/consts.ts';
import { createScenarioByCurriculumState } from '../../../../ml-common/Curriculum/createScenarioByCurriculumState.ts';
import { CurriculumState, Scenario } from '../../../../ml-common/Curriculum/types.ts';
import { getGameOverReward } from '../../Reward/calculateReward.ts';
import { agentSampleChannel, curriculumStateChannel, episodeSampleChannel, queueSizeChannel } from '../channels.ts';

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
        iteration: 0,
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
            train: random() < 0.95,
            iteration: this.curriculumState.iteration,
        });
    }

    protected afterEpisode(episode: Scenario) {
        const successRatio = episode.getSuccessRatio();
        const isReference = !episode.isTrain;
        const pilots = episode.getPilots();

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
            const finalReward = getGameOverReward(networkVersion, successRatio);
            const memoryBatch = agent.getMemoryBatch(finalReward);

            if (memoryBatch == null) return;

            agentSampleChannel.emit({
                networkVersion,
                memoryBatch,
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
