import { MasterAgent } from './MasterAgent.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
import { TANK_COUNT_SIMULATION_MAX, TANK_COUNT_SIMULATION_MIN, TICK_TRAIN_TIME } from '../../../Common/consts.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { createBattlefield } from '../../../Common/createBattlefield.ts';
import { createInputVector } from '../../../Common/createInputVector.ts';
import { applyActionToTank } from '../../../Common/applyActionToTank.ts';
import { randomRangeInt } from '../../../../../../../lib/random.ts';
import { frameTasks } from '../../../../../../../lib/TasksScheduler/frameTasks.ts';
import { CONFIG } from '../../Common/config.ts';
import { getDrawState } from '../../../Common/utils.ts';
import { EntityId } from 'bitecs';
import { calculateReward } from '../../../Common/calculateReward.ts';

export class MasterManager {
    public agent!: MasterAgent;

    private stopTrainingTimeout: VoidFunction | null = null;

    private stopGameLoopInterval: VoidFunction | null = null;
    private battlefield!: Awaited<ReturnType<typeof createBattlefield>>;
    private tankRewards = new Map<EntityId, number>();

    constructor() {

    }

    static create() {
        return new MasterManager().init();
    }

    public start() {
        this.gameLoop();
        this.trainingLoop();
    }

    public getReward(tankEid: EntityId) {
        return this.tankRewards.get(tankEid) || 0;
    }

    // Save models
    private async save() {
        return this.agent.save();
    }

    private async init() {
        this.agent = await MasterAgent.create();
        await this.agent.save();
        return this;
    }

    // Main game loop
    private async gameLoop() {
        this.stopGameLoopInterval?.();

        this.battlefield?.destroy();
        this.battlefield = await createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX), true);
        let frameCount = 0;
        let activeTanks: number[] = [];

        this.stopGameLoopInterval = frameTasks.addInterval(async () => {
            if (!getDrawState()) return;

            if (frameCount === -2) {
                frameCount = -1;
                activeTanks = [];

                frameCount = 0;
            }

            if (frameCount < 0) {
                return;
            }

            const width = GameDI.width;
            const height = GameDI.height;
            const shouldEvery = 12;
            const shouldAction = frameCount % shouldEvery === 0;
            const shouldReward = frameCount % shouldEvery === 10;
            GameDI.shouldCollectTensor = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

            if (shouldAction) {
                activeTanks = this.battlefield.getTanks();

                // Update each tank's RL controller
                for (const tankEid of activeTanks) {
                    this.updateTankBehaviour(tankEid, width, height);
                }
            }

            this.battlefield.gameTick(16.666);

            if (shouldReward) {
                for (const tankEid of activeTanks) {
                    this.tankRewards.set(
                        tankEid,
                        calculateReward(tankEid, GameDI.width, GameDI.height, frameCount).totalReward,
                    );
                }
            }


            const isEpisodeDone = activeTanks.length <= 1 || frameCount > CONFIG.maxFrames;

            if (isEpisodeDone) {
                this.gameLoop();
                return;
            }

            frameCount++;
        }, 1);
    }

    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
    ) {
        // Create input vector for the current state
        const inputVector = createInputVector(tankEid, width, height);
        // Get action from agent
        const result = this.agent.predict(inputVector);
        // Apply action to tank controller
        applyActionToTank(tankEid, result.action);
    }

    private trainingLoop() {
        this.stopTrainingTimeout = macroTasks.addTimeout(async () => {
            try {
                if (await this.agent.tryTrain()) {
                    void this.save();
                }

                this.trainingLoop();
            } catch (error) {
                this.stopTrainingTimeout?.();
                this.stopTrainingTimeout = null;

                console.error('Error during training:', error);
                macroTasks.addTimeout(() => this.trainingLoop(), TICK_TRAIN_TIME * 100);
            }
        }, TICK_TRAIN_TIME);
    }
}