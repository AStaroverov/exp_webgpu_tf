/**
 * createUnknownScenario — builds one headless training world for ppo_unknown.
 *
 * The analogue of tanks' `createScenarioGridBase` + `createScenarioCore`, but
 * collapsed to one fixed self-play N-vs-M scenario for the MVP (no curriculum
 * sampling yet). Steps:
 *   1. createGame headless (no render target).
 *   2. spawn TEAM_SIZE tanks for each of the 2 teams on distinct passable cells.
 *   3. give every learning tank a board-observation component + an UnknownAgent.
 *   4. install the policy driver as the SystemGroup.Before plugin (in place of the
 *      stand-in driver, which the base createGame no longer adds).
 *
 * Returns a Scenario the EpisodeManager drives: gameTick + termination/metrics.
 */

import { query, QueryResult } from 'bitecs';
import { createGame } from '../../../unknown/src/Game/createGame.ts';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { PluginDI } from '../../../unknown/src/Game/DI/PluginDI.ts';
import { SystemGroup } from '../../../unknown/src/Game/ECS/Plugins/systems.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { createTank } from '../../../unknown/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../../unknown/src/Game/Config/index.ts';
import { getTankHealth, getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { getTeamsCount } from '../../../unknown/src/Game/ECS/Components/TeamRef.ts';
import { TEAM_SIZE, TEAMS_COUNT } from '../consts.ts';
import { UnknownInputBoard } from '../state/board.ts';
import { scoreTracker } from '../reward/ScoreTracker.ts';
import { UnknownAgent } from './UnknownAgent.ts';
import { createPolicyDriverSystem } from './createPolicyDriverSystem.ts';

const FIELD_SIZE = 1000;

const TEAM_COLORS: Array<[number, number, number, number]> = [
    [1.0, 0.4, 0.4, 1],
    [0.4, 0.7, 1.0, 1],
];

export type Scenario = {
    world: (typeof GameDI)['world'];
    index: number;
    train: boolean;
    width: number;
    height: number;
    agents: UnknownAgent[];
    gameTick: (delta: number) => void;
    destroy: () => void;
    setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
    getVehicleEids: () => QueryResult;
    getTeamsCount: () => number;
    getSuccessRatio: () => number;
};

export function createUnknownScenario(options: { index: number; train?: boolean }): Scenario {
    const train = options.train ?? true;
    scoreTracker.reset(); // fresh combat score per episode
    const game = createGame({ width: FIELD_SIZE, height: FIELD_SIZE });
    const world = game.world;
    const { Tank, Vehicle, VehicleController } = getGameComponents(world);

    const cells = pickDistinctCells(TEAMS_COUNT * TEAM_SIZE);
    const agents: UnknownAgent[] = [];
    const agentMap = new Map<number, UnknownAgent>();

    let slot = 0;
    for (let team = 0; team < TEAMS_COUNT; team++) {
        for (let n = 0; n < TEAM_SIZE; n++) {
            const cell = cells[slot++];
            if (!cell) continue;
            const pos = MapDI.grid.hexToWorld(cell);
            if (!pos) continue;

            const tankEid = createTank({
                type: VehicleType.MediumTank,
                playerId: slot,
                teamId: team,
                x: pos.x,
                y: pos.y,
                rotation: Math.random() * Math.PI * 2,
                color: new Float32Array(TEAM_COLORS[team % TEAM_COLORS.length]),
            });
            VehicleController.setMove$(tankEid, 0);
            VehicleController.setRotate$(tankEid, 0);

            UnknownInputBoard.addComponent(world, tankEid);
            const agent = new UnknownAgent(tankEid, train);
            agents.push(agent);
            agentMap.set(tankEid, agent);
        }
    }

    PluginDI.addSystem(SystemGroup.Before, createPolicyDriverSystem(agentMap));

    // Capture initial per-team health on the first tick to base the success ratio on.
    let initialTeamHealth: Record<number, number> | undefined;
    const rawGameTick = game.gameTick;
    const gameTick = (delta: number) => {
        initialTeamHealth ??= getTeamHealth(world);
        rawGameTick(delta);
    };

    return {
        world,
        index: options.index,
        train,
        width: FIELD_SIZE,
        height: FIELD_SIZE,
        agents,
        gameTick,
        destroy: () => game.destroy(),
        setRenderTarget: (canvas) => game.setRenderTarget(canvas),
        getVehicleEids: () => query(world, [Vehicle, Tank]),
        getTeamsCount: () => getTeamsCount(),
        getSuccessRatio: () => {
            if (!initialTeamHealth) return 0;
            return computeSuccessRatio(initialTeamHealth, getTeamHealth(world));
        },
    };

    function pickDistinctCells(count: number): Array<{ q: number; r: number }> {
        const grid = MapDI.grid;
        const all: Array<{ q: number; r: number }> = [];
        grid.forEachCell((cell) => {
            if (grid.isPassable(cell.q, cell.r)) all.push({ q: cell.q, r: cell.r });
        });
        for (let i = all.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [all[i], all[j]] = [all[j], all[i]];
        }
        return all.slice(0, count);
    }
}

/** Sum of normalized tank health per team id. */
function getTeamHealth(world: (typeof GameDI)['world']): Record<number, number> {
    const { Tank } = getGameComponents(world);
    const tanks = query(world, [Tank]);
    const health: Record<number, number> = {};
    for (let i = 0; i < tanks.length; i++) {
        const eid = tanks[i];
        const team = getTankTeamId(eid);
        health[team] = (health[team] ?? 0) + getTankHealth(eid);
    }
    return health;
}

/**
 * Team-0-perspective success in [-1, 1]: relative surviving-health advantage.
 *   +1  → team 0 intact, team 1 wiped;  −1 → the reverse.
 */
function computeSuccessRatio(
    initial: Record<number, number>,
    current: Record<number, number>,
): number {
    const init0 = initial[0] ?? 0;
    const init1 = initial[1] ?? 0;
    const share0 = init0 > 0 ? (current[0] ?? 0) / init0 : 0;
    const share1 = init1 > 0 ? (current[1] ?? 0) / init1 : 0;
    const total = share0 + share1;
    if (total <= 0) return 0;
    return (share0 - share1) / total;
}
