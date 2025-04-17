import { createBattlefield } from '../../Common/createBattlefield.ts';
import { TANK_COUNT_SIMULATION_MAX, TANK_COUNT_SIMULATION_MIN, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { ActorAgent } from './ActorAgent.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';
import { CONFIG } from '../config.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { prepareInputArrays } from '../../Common/InputArrays.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { memoryChannel } from '../../DB';
import { getTankHealth } from '../../../ECS/Components/Tank/TankHealth.ts';
import { calculateReward } from '../../Reward/calculateReward.ts';

type Game = Awaited<ReturnType<typeof createBattlefield>>;

export class ActorManager {
    private agent!: ActorAgent;

    constructor() {
        this.agent = ActorAgent.create();
    }

    public static create() {
        return new ActorManager();
    }

    async start() {
        while (true) {
            try {
                await this.runEpisode();
            } catch (error) {
                console.error('Error during episode:', error);
            }
        }
    }

    private beforeEpisode() {
        return this.agent.sync().then(() => Promise.all([
            createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX)),
        ]));
    }

    private afterEpisode() {
        const memory = this.agent.readMemory();
        memoryChannel.emit(memory);
    }

    private cleanupEpisode(game: Game) {
        this.agent.dispose();
        game.destroy();
    }

    private async runEpisode() {
        const [game] = await this.beforeEpisode();

        try {
            await this.runGameLoop(game);
            this.afterEpisode();
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(game);
        }
    }

    private runGameLoop(game: Game) {
        return new Promise(resolve => {
            const shouldEvery = 12;
            const warmupFramesCount = CONFIG.warmupFrames - (CONFIG.warmupFrames % shouldEvery);
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            const width = GameDI.width;
            const height = GameDI.height;
            let frameCount = 0;
            let regardedTanks: number[] = [];

            const stop = macroTasks.addInterval(() => {
                for (let i = 0; i < 100; i++) {
                    frameCount++;

                    const currentTanks = game.getTanks();
                    const isEpisodeDone = currentTanks.length <= 1 || game.getTeamsCount() <= 1 || frameCount > maxFramesCount;

                    const isWarmup = frameCount < warmupFramesCount;
                    const shouldAction = frameCount % shouldEvery === 0;
                    const shouldMemorize = isEpisodeDone
                        || ((frameCount - 3) % shouldEvery === 0
                            || (frameCount - 7) % shouldEvery === 0
                            || (frameCount - 11) % shouldEvery === 0);
                    TenserFlowDI.shouldCollectState = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

                    if (shouldAction) {
                        regardedTanks = currentTanks;

                        for (const tankEid of regardedTanks) {
                            this.updateTankBehaviour(tankEid, width, height, isWarmup);
                        }
                    }

                    // Execute game tick
                    game.gameTick(TICK_TIME_SIMULATION * (isWarmup ? 2 : 1));

                    if (isWarmup) {
                        continue;
                    }

                    if (shouldMemorize) {
                        for (const tankEid of regardedTanks) {
                            this.memorizeTankBehaviour(
                                tankEid,
                                width,
                                height,
                                isEpisodeDone,
                            );
                        }
                    }

                    if (isEpisodeDone) {
                        stop();
                        resolve(null);
                        break;
                    }
                }
            }, 1);
        });
    }

    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        isWarmup: boolean,
    ) {
        // Create input vector for the current state
        const state = prepareInputArrays(tankEid, width, height);
        // Get action from agent
        const result = this.agent.act(state);
        // Apply action to tank controller
        applyActionToTank(tankEid, result.actions);

        if (!isWarmup) {
            const stateReward = calculateReward(
                tankEid,
                width,
                height,
                false,
            );

            this.agent.rememberAction(
                tankEid,
                state,
                stateReward,
                result.actions,
                result.mean,
                result.logStd,
                result.logProb,
            );
        }
    }

    private memorizeTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        isEpisodeDone: boolean,
    ) {
        // Calculate reward
        const reward = calculateReward(
            tankEid,
            width,
            height,
            isEpisodeDone,
        );
        const isDone = isEpisodeDone || getTankHealth(tankEid) <= 0;

        // Store experience in agent's memory
        this.agent.rememberReward(
            tankEid,
            reward,
            isDone,
        );
    }
}
