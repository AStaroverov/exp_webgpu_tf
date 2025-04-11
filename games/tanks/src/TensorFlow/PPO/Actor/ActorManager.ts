import { createBattlefield } from '../../Common/createBattlefield.ts';
import { TANK_COUNT_SIMULATION_MAX, TANK_COUNT_SIMULATION_MIN, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { ActorAgent } from './ActorAgent.ts';
import { calculateReward } from '../../Common/calculateReward.ts';
import { getTankHealth } from '../../../ECS/Components/Tank.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';
import { CONFIG } from '../config.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { prepareInputArrays } from '../../Common/InputArrays.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { memoryChannel } from '../../DB';

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
                await new Promise(resolve => macroTasks.addTimeout(resolve, 100));
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
            this.runGameLoop(game);
            this.afterEpisode();
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(game);
        }
    }

    private runGameLoop(game: Game) {
        const shouldEvery = 12;
        const maxWarmupFrames = CONFIG.warmupFrames - (CONFIG.warmupFrames % shouldEvery);
        const maxEpisodeFrames = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
        const width = GameDI.width;
        const height = GameDI.height;
        let frameCount = 0;
        let activeTanks: number[] = [];

        for (let i = 0; i <= maxEpisodeFrames; i++) {
            frameCount++;

            const isWarmup = frameCount < maxWarmupFrames;
            const shouldAction = frameCount % shouldEvery === 0;
            const shouldMemorize =
                (frameCount - 4) % shouldEvery === 0
                || (frameCount - 7) % shouldEvery === 0
                || (frameCount - 10) % shouldEvery === 0;
            const isLastMemorize = frameCount > 10 && (frameCount - 10) % shouldEvery === 0;
            TenserFlowDI.shouldCollectState = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

            if (shouldAction) {
                activeTanks = game.getTanks();

                for (const tankEid of activeTanks) {
                    this.updateTankBehaviour(tankEid, width, height, isWarmup);
                }
            }

            // Execute game tick
            game.gameTick(TICK_TIME_SIMULATION * (isWarmup ? 2 : 1));

            if (isWarmup) {
                continue;
            }

            if (shouldMemorize) {
                for (const tankEid of activeTanks) {
                    this.memorizeTankBehaviour(tankEid, width, height, frameCount, isLastMemorize ? 0.5 : 0.25);
                }
            }

            if (activeTanks.length <= 1) {
                break;
            }
        }
    }

    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        isWarmup: boolean,
    ) {
        // Create input vector for the current state
        const input = prepareInputArrays(tankEid, width, height);
        // Get action from agent
        const result = this.agent.act(input);
        // Apply action to tank controller
        applyActionToTank(tankEid, result.actions);

        if (!isWarmup) {
            this.agent.rememberAction(
                tankEid,
                input,
                result.actions,
                result.logProb,
            );
        }
    }

    private memorizeTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        step: number,
        rewardMultiplier: number,
    ) {
        // Calculate reward
        const reward = calculateReward(
            tankEid,
            width,
            height,
            step,
        ).totalReward;
        // Check if tank is "dead" based on health
        const isDone = getTankHealth(tankEid) <= 0;

        // Store experience in agent's memory
        this.agent.rememberReward(
            tankEid,
            reward * rewardMultiplier,
            isDone,
        );
    }
}
