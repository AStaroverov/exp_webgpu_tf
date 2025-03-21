import { MasterAgent } from './MasterAgent.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
import {
    MAX_FRAMES,
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TRAIN_TIME,
} from '../../../Common/consts.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { createBattlefield } from '../../../Common/createBattlefield.ts';
import { createInputVector } from '../../../Common/createInputVector.ts';
import { applyActionToTank } from '../../../Common/applyActionToTank.ts';
import { randomRangeInt } from '../../../../../../../lib/random.ts';
import { frameTasks } from '../../../../../../../lib/TasksScheduler/frameTasks.ts';

export class MasterManager {
    public agent!: MasterAgent;

    private stopTrainingTimeout: VoidFunction | null = null;

    private stopGameLoopInterval: VoidFunction | null = null;
    private frameCount = -2;
    private battlefield!: Awaited<ReturnType<typeof createBattlefield>>;

    constructor() {

    }

    static create() {
        return new MasterManager().init();
    }

    public start() {
        this.gameLoop();
        this.trainingLoop();
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

        this.frameCount = 0;
        this.battlefield?.destroy();
        this.battlefield = await createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX), true);
        let activeTanks: number[] = [];

        this.stopGameLoopInterval = frameTasks.addInterval(async () => {
            try {
                if (this.frameCount === -2) {
                    this.frameCount = -1;
                    activeTanks = [];

                    this.frameCount = 0;
                }
                if (this.frameCount < 0) {
                    return;
                }

                const width = GameDI.width;
                const height = GameDI.height;
                const shouldEvery = 12;
                const shouldAction = this.frameCount % shouldEvery === 0;
                GameDI.shouldCollectTensor = this.frameCount > 0 && (this.frameCount + 1) % shouldEvery === 0;

                if (shouldAction) {
                    activeTanks = this.battlefield.getTanks();

                    // Update each tank's RL controller
                    for (const tankEid of activeTanks) {
                        this.updateTankBehaviour(tankEid, width, height);
                    }
                }

                this.battlefield.gameTick(16.666);

                this.frameCount++;

                const isEpisodeDone = activeTanks.length <= 1 || this.frameCount > MAX_FRAMES;

                if (isEpisodeDone) {
                    this.gameLoop();
                }
            } catch (error) {
                console.error('Error during game loop:', error);
                this.stopGameLoopInterval?.();
                this.stopGameLoopInterval = null;
                macroTasks.addTimeout(() => this.gameLoop(), 1000);
            }
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
                const trainsCount = await this.agent.tryTrain();

                if (trainsCount > 0) {
                    await this.save();
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