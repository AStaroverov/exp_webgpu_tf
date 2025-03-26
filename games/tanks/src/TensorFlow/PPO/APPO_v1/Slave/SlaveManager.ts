import { createBattlefield } from '../../../Common/createBattlefield.ts';
import {
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_SIMULATION,
} from '../../../Common/consts.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../../lib/random.ts';
import { SlaveAgent } from './SlaveAgent.ts';
import { createInputVector } from '../../../Common/createInputVector.ts';
import { calculateReward } from '../../../Common/calculateReward.ts';
import { getTankHealth } from '../../../../ECS/Components/Tank.ts';
import { applyActionToTank } from '../../../Common/applyActionToTank.ts';
import { addMemoryBatch } from '../Database.ts';
import { CONFIG } from '../../Common/config.ts';

export class SlaveManager {
    private agent!: SlaveAgent;

    private battlefield: Awaited<ReturnType<typeof createBattlefield>> | null = null;
    private tankRewards = new Map<number, number>();

    constructor() {
        this.agent = SlaveAgent.create();
    }

    public static create() {
        return new SlaveManager();
    }

    start() {
        this.trainingLoop();
    }

    dispose() {
        this.tankRewards.clear();
        this.battlefield?.destroy();
        this.agent.dispose();
    }

    // Initialize the game environment
    async init() {
        [this.battlefield] = await Promise.all([
            createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX)),
            this.agent.sync(),
        ]);
    }

    // Main game loop
    private async trainingLoop() {
        await this.init();

        let frameCount = 0;

        while (true) {
            await Promise.resolve();
            // play
            frameCount++;
            const width = GameDI.width;
            const height = GameDI.height;
            const shouldEvery = 12;
            const isWarmup = frameCount < shouldEvery * 8;
            const shouldAction = frameCount % shouldEvery === 0;
            const shouldMemorize =
                (frameCount - 4) % shouldEvery === 0
                || (frameCount - 7) % shouldEvery === 0
                || (frameCount - 10) % shouldEvery === 0;
            const isLastMemorize = frameCount > 10 && (frameCount - 10) % shouldEvery === 0;
            GameDI.shouldCollectTensor = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

            const activeTanks = this.battlefield!.getTanks();

            if (shouldAction) {
                for (const tankEid of activeTanks) {
                    this.updateTankBehaviour(tankEid, width, height, isWarmup);
                }
            }

            // Execute game tick
            this.battlefield!.gameTick(TICK_TIME_SIMULATION);

            if (isWarmup) {
                continue;
            }

            if (shouldMemorize) {
                for (const tankEid of activeTanks) {
                    this.memorizeTankBehaviour(tankEid, width, height, frameCount, isLastMemorize ? 0.5 : 0.25, isLastMemorize);
                }
            }

            const isEpisodeDone = activeTanks.length <= 1 || frameCount > CONFIG.maxFrames;

            if (isEpisodeDone) {
                break;
            }
        }

        this.exposeMemory();
        this.dispose();
        this.trainingLoop();
    }

    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        isWarmup: boolean,
    ) {
        // Create input vector for the current state
        const inputVector = createInputVector(tankEid, width, height);
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

        // Accumulate reward for this tank
        this.tankRewards.set(tankEid, (this.tankRewards.get(tankEid) || 0) + reward);

        // Store experience in agent's memory
        this.agent.rememberReward(
            tankEid,
            reward * rewardMultiplier,
            isDone,
            isLast,
        );
    }

    private async exposeMemory() {
        return addMemoryBatch(this.agent.readMemory());
    }
}
