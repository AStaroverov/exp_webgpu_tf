import { createBattlefield } from '../../Common/createBattlefield.ts';
import {
    SNAPSHOT_EVERY,
    TANK_COUNT_SIMULATION_MAX,
    TANK_COUNT_SIMULATION_MIN,
    TICK_TIME_SIMULATION,
} from '../../Common/consts.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';
import { ActorAgent } from './ActorAgent.ts';
import { applyActionToTank } from '../../Common/applyActionToTank.ts';
import { CONFIG } from '../config.ts';
import { macroTasks } from '../../../../../../lib/TasksScheduler/macroTasks.ts';
import { prepareInputArrays } from '../../Common/InputArrays.ts';
import { TenserFlowDI } from '../../../DI/TenserFlowDI.ts';
import { actorMemoryChannel } from '../channels.ts';
import { calculateReward } from '../../Reward/calculateReward.ts';
import { getTankHealth, getTankTeamId } from '../../../ECS/Entities/Tank/TankUtils.ts';

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

    private afterEpisode(successRatio: number) {
        const memory = this.agent.readMemory();
        actorMemoryChannel.emit({ ...memory, successRatio });
    }

    private cleanupEpisode(game: Game) {
        this.agent.dispose();
        game.destroy();
    }

    private async runEpisode() {
        const [game] = await this.beforeEpisode();
        const tanks = game.getTanks();

        const teamHealth = getTeamHealth(tanks);

        try {
            await this.runGameLoop(game);

            const successRatio = getSuccessRatio(game, teamHealth);

            this.afterEpisode(successRatio);
        } catch (error) {
            throw error;
        } finally {
            this.cleanupEpisode(game);
        }
    }

    private runGameLoop(game: Game) {
        return new Promise(resolve => {
            const shouldEvery = SNAPSHOT_EVERY;
            const warmupFramesCount = CONFIG.warmupFrames - (CONFIG.warmupFrames % shouldEvery);
            const maxFramesCount = (CONFIG.episodeFrames - (CONFIG.episodeFrames % shouldEvery) + shouldEvery);
            const width = GameDI.width;
            const height = GameDI.height;
            let regardedTanks: number[] = [];
            let frame = 0;

            const stop = macroTasks.addInterval(() => {
                for (let i = 0; i < 100; i++) {
                    frame++;

                    const agentTanks = game.getAgenTanks();
                    const currentTanks = game.getTanks();
                    const gameOverByTankCount = agentTanks.length <= 0 || currentTanks.length <= 1;
                    const gameOverByTeamWin = game.getTeamsCount() === 1;
                    const gameOverByTime = frame > maxFramesCount;
                    const gameOver = gameOverByTankCount || gameOverByTeamWin || gameOverByTime;

                    const isWarmup = frame < warmupFramesCount;
                    const shouldAction = frame % shouldEvery === 0;
                    const shouldMemorize = gameOver || (frame - (shouldEvery - 1)) % shouldEvery === 0;
                    TenserFlowDI.shouldCollectState = (frame + 1) % shouldEvery === 0;

                    if (shouldAction) {
                        regardedTanks = agentTanks;

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
                                gameOver,
                            );
                        }
                    }

                    if (gameOver) {
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
        gameOver: boolean,
    ) {
        const isDead = getTankHealth(tankEid) <= 0;
        const isDone = gameOver || isDead;
        const reward = calculateReward(
            tankEid,
            width,
            height,
        );

        this.agent.rememberReward(
            tankEid,
            reward,
            isDone,
        );
    }
}

function getTeamHealth(tanks: number[]) {
    return tanks.reduce((acc, tankEid) => {
        const team = getTankTeamId(tankEid);
        const health = getTankHealth(tankEid);
        acc[team] = (acc[team] || 0) + health;
        return acc;
    }, {} as Record<number, number>);
}

function getSuccessRatio(game: Game, initialTeamHealth: Record<number, number>) {
    const activeTeam = game.activeTeam;
    const tanks = game.getTanks();
    const teamHealth = getTeamHealth(tanks);
    const successRatio = Object.entries(teamHealth)
        .map(([k, v]) => [Number(k), v])
        .reduce((acc, [teamId, health]) => {
            const delta = activeTeam === Number(teamId)
                ? (health / initialTeamHealth[teamId])
                : 1 - (health / initialTeamHealth[teamId]);
            return acc + delta;
        }, 0);

    return successRatio / game.getTeamsCount();
}