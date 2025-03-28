import { createBattlefield } from '../../../Common/createBattlefield.ts';
import { TANK_COUNT_SIMULATION_MAX, TANK_COUNT_SIMULATION_MIN, TICK_TIME_SIMULATION } from '../../../Common/consts.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../../lib/random.ts';
import { SlaveAgent } from './SlaveAgent.ts';
import { calculateReward } from '../../../Common/calculateReward.ts';
import { getTankHealth } from '../../../../ECS/Components/Tank.ts';
import { applyActionToTank } from '../../../Common/applyActionToTank.ts';
import { addMemoryBatch } from '../Database.ts';
import { CONFIG } from '../../Common/config.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
import { prepareInputArrays } from '../../../Common/prepareInputArrays.ts';

export class SlaveManager {
    private agent!: SlaveAgent;

    constructor() {
        this.agent = SlaveAgent.create();
    }

    public static create() {
        return new SlaveManager();
    }

    start() {
        this.trainingLoop();
    }

    private beforeTraining() {
        return Promise.all([
            createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX)),
            this.agent.sync(),
        ]);
    }

    private afterTraining(game: { destroy: () => void }) {
        addMemoryBatch(this.agent.readMemory());
        this.agent.dispose();
        game.destroy();
    }

    // Main game loop
    private async trainingLoop() {
        const [game] = await this.beforeTraining();

        const shouldEvery = 12;
        const maxWarmupFrames = CONFIG.warmupFrames - (CONFIG.warmupFrames % shouldEvery);
        const maxEpisodeFrames = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
        const width = GameDI.width;
        const height = GameDI.height;
        let frameCount = 0;

        for (let i = 0; i <= maxEpisodeFrames; i++) {
            if (frameCount % 100 === 0) {
                await new Promise(resolve => macroTasks.addTimeout(resolve, 0));
            }

            frameCount++;

            const isWarmup = frameCount < maxWarmupFrames;
            const shouldAction = frameCount % shouldEvery === 0;
            const shouldMemorize =
                (frameCount - 4) % shouldEvery === 0
                || (frameCount - 7) % shouldEvery === 0
                || (frameCount - 10) % shouldEvery === 0;
            const isLastMemorize = frameCount > 10 && (frameCount - 10) % shouldEvery === 0;
            GameDI.shouldCollectTensor = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

            const activeTanks = game.getTanks();

            if (shouldAction) {
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
                    this.memorizeTankBehaviour(tankEid, width, height, frameCount, isLastMemorize ? 0.5 : 0.25, isLastMemorize);
                }
            }

            if (activeTanks.length <= 1) {
                break;
            }
        }

        this.afterTraining(game);
        this.trainingLoop();
    }


    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        isWarmup: boolean,
    ) {
        // Create input vector for the current state
        const inputVector = prepareInputArrays(tankEid, width, height);
        // Get action from agent
        const result = this.agent.act(inputVector);
        // Apply action to tank controller
        applyActionToTank(tankEid, result.actions);

        if (!isWarmup) {
            this.agent.rememberAction(
                tankEid,
                inputVector,
                result.actions,
                result.logProb,
                result.value,
            );
        }
    }

    private memorizeTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        step: number,
        rewardMultiplier: number,
        isLast: boolean,
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
            isLast,
        );
    }
}
