import { PlayerAgent } from './PlayerAgent.ts';
import { TANK_COUNT_SIMULATION_MAX, TANK_COUNT_SIMULATION_MIN, TICK_TIME_SIMULATION } from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { createBattlefield } from '../../Common/createBattlefield.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { CONFIG } from '../config.ts';
import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { calculateReward } from '../../Common/calculateReward.ts';
import { prepareInputArrays } from '../../Common/prepareInputArrays.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';

export class PlayerManager {
    public agent!: PlayerAgent;

    private stopGameLoopInterval: VoidFunction | null = null;
    private battlefield!: Awaited<ReturnType<typeof createBattlefield>>;
    private tankRewards = new Map<EntityId, number>();

    constructor() {

    }

    static create() {
        return new PlayerManager().init();
    }

    public start() {
        this.gameLoop();
    }

    public getReward(tankEid: EntityId) {
        return this.tankRewards.get(tankEid) || 0;
    }

    private async init() {
        this.agent = await PlayerAgent.create();
        return this;
    }

    // Main game loop
    private async gameLoop() {
        this.stopGameLoopInterval?.();

        this.battlefield?.destroy();
        this.battlefield = await createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX), true);
        await this.agent.sync();

        const shouldEvery = 12;
        const maxEpisodeFrames = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
        const width = GameDI.width;
        const height = GameDI.height;

        let frameCount = 0;
        let activeTanks: number[] = [];

        this.stopGameLoopInterval = frameTasks.addInterval(async () => {
            if (!getDrawState()) {
                frameCount = -1;
                return;
            }

            if (frameCount === -1) {
                this.gameLoop();
                return;
            }

            const shouldAction = frameCount > 0 && frameCount % shouldEvery === 0;
            const shouldReward = frameCount > 0 && frameCount % shouldEvery === 10;
            TenserFlowDI.shouldCollectState = frameCount > 0 && (frameCount + 1) % shouldEvery === 0;

            if (shouldAction) {
                activeTanks = this.battlefield.getTanks();

                // Update each tank's RL controller
                for (const tankEid of activeTanks) {
                    this.updateTankBehaviour(tankEid, width, height);
                }
            }

            this.battlefield.gameTick(TICK_TIME_SIMULATION);

            if (shouldReward) {
                for (const tankEid of activeTanks) {
                    this.tankRewards.set(
                        tankEid,
                        calculateReward(tankEid, GameDI.width, GameDI.height, frameCount).totalReward,
                    );
                }
            }

            frameCount++;

            const isEpisodeDone = frameCount > shouldEvery && (activeTanks.length <= 1 || frameCount > maxEpisodeFrames);

            if (isEpisodeDone) {
                frameCount = -1;
            }
        }, 1);
    }

    private updateTankBehaviour(
        tankEid: number,
        width: number,
        height: number,
    ) {
        // Create input vector for the current state
        const input = prepareInputArrays(tankEid, width, height);
        // Get action from agent
        const result = this.agent.predict(input);
        // Apply action to tank controller
        applyActionToTank(tankEid, result.actions);
    }
}