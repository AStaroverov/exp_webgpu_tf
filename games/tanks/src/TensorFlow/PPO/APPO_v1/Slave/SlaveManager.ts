import { createBattlefield } from '../../../Common/createBattlefield.ts';
import {
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_REAL,
    TICK_TIME_SIMULATION,
} from '../../../Common/consts.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
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
    private frameCount: number = -10;
    private stopTrainingLoop: VoidFunction | null = null;

    private tankRewards = new Map<number, number>();

    constructor() {
    }

    public static create() {
        return new SlaveManager().init();
    }

    dispose() {
        this.frameCount = 0;
        this.tankRewards.clear();
        this.stopTrainingLoop?.();
        this.battlefield?.destroy();
        this.agent.dispose();
    }

    // Initialize the game environment
    async init() {
        this.agent = SlaveAgent.create();
        this.trainingLoop();
    }

    private async reset() {
        this.battlefield = await createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX));
        await this.agent.sync();
    }

    // Main game loop
    private trainingLoop() {
        let state: number = 0;
        this.stopTrainingLoop = macroTasks.addInterval(async () => {
            // begin
            if (state === 0) {
                state = 1;
                await this.reset();
                state = -1;
                return;
            }

            // expose memory
            if (state === 2) {
                state = 3;
                await this.exposeMemory();
                await this.agent.dispose();
                await this.agent.waitNewVersion();
                await this.agent.sync();
                state = 4;
                return;
            }

            // finish
            if (state === 4) {
                state = 5;
                this.dispose();
                this.trainingLoop();
                return;
            }

            if (state !== -1) return;

            // play
            this.frameCount++;
            const width = GameDI.width;
            const height = GameDI.height;
            const shouldEvery = 12;
            const isWarmup = this.frameCount < shouldEvery * 8;
            const shouldAction = this.frameCount % shouldEvery === 0;
            const shouldMemorize =
                (this.frameCount - 4) % shouldEvery === 0
                || (this.frameCount - 7) % shouldEvery === 0
                || (this.frameCount - 10) % shouldEvery === 0;
            const isLastMemorize = this.frameCount > 10 && (this.frameCount - 10) % shouldEvery === 0;
            GameDI.shouldCollectTensor = this.frameCount > 0 && (this.frameCount + 1) % shouldEvery === 0;

            const activeTanks = this.battlefield!.getTanks();

            if (shouldAction) {
                for (const tankEid of activeTanks) {
                    this.updateTankBehaviour(tankEid, width, height, isWarmup);
                }
            }

            // Execute game tick
            this.battlefield!.gameTick(TICK_TIME_SIMULATION);

            if (isWarmup) {
                return;
            }

            if (shouldMemorize) {
                for (const tankEid of activeTanks) {
                    this.memorizeTankBehaviour(tankEid, width, height, this.frameCount, isLastMemorize ? 0.5 : 0.25, isLastMemorize);
                }
            }

            const isEpisodeDone = activeTanks.length <= 1 || this.frameCount > CONFIG.maxFrames;

            if (isEpisodeDone) {
                state = 2;
                return;
            }
        }, TICK_TIME_REAL);
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
                result.rawActions,
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
        const data = this.agent.readMemory();
        await addMemoryBatch(data);
    }
}
