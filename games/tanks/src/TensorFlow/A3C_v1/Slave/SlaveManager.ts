import { createBattlefield } from '../../Common/createBattlefield.ts';
import {
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_REAL,
    TICK_TIME_SIMULATION,
} from '../../Common/consts.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { SlaveAgent } from './SlaveAgent.ts';
import { createInputVector } from '../../Common/createInputVector.ts';
import { calculateReward } from '../../Common/calculateReward.ts';
import { getTankHealth } from '../../../ECS/Components/Tank.ts';
import { addGradients } from '../Database.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';


export class SlaveManager {
    private agent!: SlaveAgent;

    private battlefield: Awaited<ReturnType<typeof createBattlefield>> | null = null;
    private frameCount: number = -10;
    private episodeCount: number = 0;
    private stopFrameInterval: VoidFunction | null = null;

    private tankRewards = new Map<number, number>();

    constructor() {
    }

    public static create() {
        return new SlaveManager().init();
    }

    dispose() {
        this.tankRewards.clear();
        this.battlefield?.destroy();
    }

    // Initialize the game environment
    async init() {
        this.agent = SlaveAgent.create();
        this.trainingLoop();
    }

    // Reset environment for a new episode
    private async resetEnvironment() {
        this.dispose();
        await this.resetAgent();

        this.battlefield = await createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX));

        // Reset frame counter
        this.frameCount = 0;
        // Increment episode counter
        this.episodeCount++;

        console.log(`Environment reset for episode ${ this.episodeCount }`);

        return this.battlefield;
    }

    private async resetAgent() {
        this.agent.dispose();
        await this.agent.load();
    }

    // Main game loop
    private async trainingLoop() {
        let skipTick = false;
        let activeTanks: number[] = [];

        await this.resetEnvironment();

        this.stopFrameInterval = macroTasks.addInterval(async () => {
            if (skipTick) return;

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

            if (shouldAction) {
                activeTanks = this.battlefield!.getTanks();
                // const destroyedTanks = prevActiveTanks.filter(tankEid => !activeTanks.includes(tankEid));

                // Update each tank's RL controller
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
                    this.memorizeTankBehaviour(tankEid, width, height, this.episodeCount, isLastMemorize ? 0.5 : 0.25, isLastMemorize);
                }

                if (isLastMemorize && this.agent.isReady()) {
                    skipTick = true;
                    await this.exposeGradients();
                    await this.resetAgent();
                    skipTick = false;
                }
            }

            // Check if episode is done
            const isEpisodeDone = activeTanks.length <= 1 || this.frameCount > 1500;

            if (isEpisodeDone) {
                this.stopFrameInterval?.();
                this.trainingLoop();
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
                result.value,
            );
        }
    }

    private memorizeTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
        episode: number,
        rewardMultiplier: number,
        isLast: boolean,
    ) {

        // Calculate reward
        const reward = calculateReward(
            tankEid,
            width,
            height,
            episode,
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

    private exposeGradients() {
        const grads = this.agent.computeGradients();

        return addGradients(grads);
    }
}
