import { PlayerAgent } from './PlayerAgent.ts';
import {
    SNAPSHOT_EVERY,
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_SIMULATION,
} from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { createBattlefield } from '../../Common/createBattlefield.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { CONFIG } from '../config.ts';
import { getDrawState } from '../../Common/uiUtils.ts';
import { EntityId } from 'bitecs';
import { prepareInputArrays } from '../../Common/InputArrays.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { first, firstValueFrom, interval } from 'rxjs';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { frameTasks } from '../../../../../../lib/TasksScheduler/frameTasks.ts';
import { calculateReward } from '../../Reward/calculateReward.ts';

type Game = Awaited<ReturnType<typeof createBattlefield>>;

export class PlayerManager {
    public agent!: PlayerAgent;

    private tankRewards = new Map<EntityId, number>();

    constructor() {

    }

    static create() {
        return new PlayerManager().init();
    }

    public async start() {
        while (true) {
            try {
                await this.waitEnabling();
                await this.runEpisode();
            } catch (error) {
                console.error('Error during episode:', error);
                await new Promise(resolve => macroTasks.addTimeout(resolve, 1000));
            }
        }
    }

    public getReward(tankEid: EntityId) {
        return this.tankRewards.get(tankEid) || 0;
    }

    private async init() {
        this.agent = PlayerAgent.create();
        return this;
    }

    private waitEnabling() {
        return firstValueFrom(interval(1000).pipe(
            first(getDrawState),
        ));
    }

    private beforeEpisode() {
        return Promise.all([
            createBattlefield(randomRangeInt(TANK_COUNT_SIMULATION_MIN, TANK_COUNT_SIMULATION_MAX), true),
            this.agent.sync(),
        ]);
    }

    private cleanupEpisode(game: Game) {
        game.destroy();
    }

    private async runEpisode() {
        const [game] = await this.beforeEpisode();

        try {
            await this.runGameLoop(game);
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(game);
        }
    }

    private async runGameLoop(game: Game) {
        return new Promise(resolve => {
            const shouldEvery = SNAPSHOT_EVERY;
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            const width = GameDI.width;
            const height = GameDI.height;
            let regardedTanks: number[] = [];
            let frame = 0;

            const stop = frameTasks.addInterval(() => {
                frame++;

                const agentTanks = game.getAgenTanks();
                const currentTanks = game.getTanks();
                const isEpisodeDone =
                    currentTanks.length <= 1
                    || agentTanks.length <= 0
                    || game.getTeamsCount() <= 1
                    || frame > maxFramesCount;

                const shouldAction = frame % shouldEvery === 0;
                const shouldReward = isEpisodeDone || ((frame - (shouldEvery - 1)) % shouldEvery === 0);
                TenserFlowDI.shouldCollectState = (frame + 1) % shouldEvery === 0;

                if (shouldAction) {
                    regardedTanks = agentTanks;

                    // Update each tank's RL controller
                    for (const tankEid of regardedTanks) {
                        this.updateTankBehaviour(tankEid, width, height);
                    }
                }

                game.gameTick(TICK_TIME_SIMULATION);

                if (shouldReward) {
                    for (const tankEid of regardedTanks) {
                        this.tankRewards.set(
                            tankEid,
                            calculateReward(tankEid, GameDI.width, GameDI.height),
                        );
                    }
                }

                if (isEpisodeDone || !getDrawState()) {
                    stop();
                    resolve(null);
                }
            }, 1);
        });
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