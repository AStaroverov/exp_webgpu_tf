/**
 * createUnknownScenario — builds one headless training world for ppo_unknown.
 *
 * The analogue of tanks' `createScenarioGridBase` + `createScenarioCore`: builds the
 * world a `ScenarioConfig` describes — team sizes plus how the enemy team (team 1)
 * behaves (see `scenarioCompositions`). Team 0 is always the learning policy. Steps:
 *   1. createGame headless (no render target).
 *   2. spawn `allies` + `enemies` units on distinct passable cells: each team of 2+
 *      gets one Ranger scout (searchlight, no gun — exercises the spotting channels),
 *      the rest a random class (Light/Medium/Heavy — class is read from the units vector).
 *   3. drive team 0 (and team 1 under self-play) with a learning UnknownAgent;
 *      drive standing/moving enemies with a scripted RandomBot (both fire
 *      sporadically, standing just doesn't move); drive frozen enemies with a
 *      FrozenAgent (historical policy snapshot, no learning).
 *   4. install the policy driver as the SystemGroup.Before plugin (in place of the
 *      stand-in driver, which the base createGame no longer adds).
 *
 * Returns a Scenario the EpisodeManager drives: gameTick + termination/metrics.
 */

import { query, QueryResult } from 'bitecs';
import { randomRangeInt } from '../../../../lib/random.ts';
import { createGame } from '../../../unknown/src/Game/createGame.ts';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { PluginDI } from '../../../unknown/src/Game/DI/PluginDI.ts';
import { SystemGroup } from '../../../unknown/src/Game/ECS/Plugins/systems.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { createTank } from '../../../unknown/src/Game/ECS/Entities/Tank/createTank.ts';
import { createRanger } from '../../../unknown/src/Game/ECS/Entities/Tank/Ranger/Ranger.ts';
import { spawnObstacles } from '../../../unknown/src/Game/ECS/Entities/Obstacle/spawnObstacles.ts';
import { VehicleType } from '../../../unknown/src/Game/Config/index.ts';
import { getTankHealth, getTankTeamId } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { getTeamsCount } from '../../../unknown/src/Game/ECS/Components/TeamRef.ts';
import { UnknownInputBoard } from '../state/board.ts';
import { scoreTracker } from '../reward/ScoreTracker.ts';
import { ScenarioConfig } from '../curriculum/types.ts';
import { UnknownAgent } from './UnknownAgent.ts';
import { FrozenAgent } from './FrozenAgent.ts';
import { RandomBot } from './RandomBot.ts';
import { createPolicyDriverSystem, TankDriver } from './createPolicyDriverSystem.ts';

const FIELD_SIZE = 1000;

// Every spawn rolls a random tank class (tanks_ml convention): type differences
// (speed, turret, reload, size) come for free from the per-type configs.
const TANK_TYPES = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

// RandomBot tuning per enemy behaviour. Both now return sporadic undirected fire
// (a random allowed direction) so the learner meets incoming rounds from rung 0;
// they differ only in how much they move. 'standing' holds position (moveProb 0)
// and fires rarely; 'moving' wanders and fires a bit more.
const STANDING_BOT = { moveProb: 0, fireProb: 0.1 };
const MOVING_BOT = { moveProb: 0.3, fireProb: 0.2 };

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
    /** Fraction of `teamId`'s initial health destroyed, in [0, 1]. */
    getTeamDestroyedRatio: (teamId: number) => number;
};

export function createUnknownScenario(options: {
    index: number;
    train?: boolean;
    /** Team sizes and enemy behaviour — one entry of `scenarioCompositions`. */
    config: ScenarioConfig;
}): Scenario {
    const train = options.train ?? true;
    const { allies, enemies, enemy } = options.config;
    scoreTracker.reset(); // fresh combat score per episode
    const game = createGame({ width: FIELD_SIZE, height: FIELD_SIZE });
    const world = game.world;
    const { Tank, Vehicle, VehicleController } = getGameComponents(world);

    spawnObstacles();

    const teamSizes = [allies, enemies];
    const borderSpawn = enemy === 'frozen' || enemy === 'self-play';
    const teamCells = borderSpawn
        ? pickBorderCells(teamSizes)
        : splitCells(pickDistinctCells(allies + enemies), teamSizes);
    const agents: UnknownAgent[] = [];
    const driverMap = new Map<number, TankDriver>();

    let playerId = 0;
    for (let team = 0; team < teamSizes.length; team++) {
        for (let n = 0; n < teamSizes[team]; n++) {
            const cell = teamCells[team][n];
            if (!cell) continue;
            playerId++;
            const pos = MapDI.grid.hexToWorld(cell.q, cell.r);
            if (!pos) continue;

            const isRanger = n === 0 && teamSizes[team] >= 2;
            const spawn = {
                playerId,
                teamId: team,
                x: pos.x,
                y: pos.y,
                rotation: Math.random() * Math.PI * 2,
                color: new Float32Array(TEAM_COLORS[team % TEAM_COLORS.length]),
            };
            const tankEid = isRanger
                ? createRanger(spawn)
                : createTank({ ...spawn, type: TANK_TYPES[randomRangeInt(0, TANK_TYPES.length - 1)] });
            VehicleController.setMove$(tankEid, 0);
            VehicleController.setRotate$(tankEid, 0);

            const isEnemy = team !== 0;
            if (isEnemy && enemy === 'standing') {
                // Holds position but occasionally fires (a gunless Ranger just no-ops it).
                driverMap.set(tankEid, new RandomBot(tankEid, STANDING_BOT));
                continue;
            }
            if (isEnemy && enemy === 'moving') {
                driverMap.set(tankEid, new RandomBot(tankEid, MOVING_BOT));
                continue;
            }
            if (isEnemy && enemy === 'frozen') {
                UnknownInputBoard.addComponent(world, tankEid);
                driverMap.set(tankEid, new FrozenAgent(tankEid));
                continue;
            }

            UnknownInputBoard.addComponent(world, tankEid);
            const agent = new UnknownAgent(tankEid, train);
            agents.push(agent);
            driverMap.set(tankEid, agent);
        }
    }

    PluginDI.addSystem(SystemGroup.Before, createPolicyDriverSystem(driverMap));

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
        getTeamDestroyedRatio: (teamId) => {
            const init = initialTeamHealth?.[teamId] ?? 0;
            if (init <= 0) return 0;
            const current = getTeamHealth(world)[teamId] ?? 0;
            return Math.min(1, Math.max(0, 1 - current / init));
        },
    };

    function pickDistinctCells(count: number): Array<{ q: number; r: number }> {
        const grid = MapDI.grid;
        const all: Array<{ q: number; r: number }> = [];
        grid.forEachCell((cell) => {
            if (grid.isPassable(cell.q, cell.r)) all.push({ q: cell.q, r: cell.r });
        });
        shuffle(all);
        return all.slice(0, count);
    }

    function pickBorderCells(sizes: number[]): Array<Array<{ q: number; r: number }>> {
        const grid = MapDI.grid;
        const passable: Array<{ q: number; r: number; x: number }> = [];
        grid.forEachCell((cell) => {
            if (!grid.isPassable(cell.q, cell.r)) return;
            const pos = grid.hexToWorld(cell.q, cell.r);
            if (pos) passable.push({ q: cell.q, r: cell.r, x: pos.x });
        });
        if (passable.length === 0) return sizes.map(() => []);

        let minX = Infinity, maxX = -Infinity;
        for (const c of passable) {
            if (c.x < minX) minX = c.x;
            if (c.x > maxX) maxX = c.x;
        }
        const band = (maxX - minX) * 0.25;
        const byX = [...passable].sort((a, b) => a.x - b.x); // left → right

        const takeSide = (count: number, fromLeft: boolean) => {
            const inBand = fromLeft
                ? byX.filter((c) => c.x <= minX + band)
                : byX.filter((c) => c.x >= maxX - band);
            // Enough free cells in the band → spread along the border; otherwise take
            // the N cells nearest this edge.
            const pool = inBand.length >= count
                ? (shuffle(inBand), inBand)
                : fromLeft ? byX.slice(0, count) : byX.slice(-count);
            return pool.slice(0, count).map((c) => ({ q: c.q, r: c.r }));
        };

        // Team 0 → left, every other team → right.
        return sizes.map((count, team) => takeSide(count, team === 0));
    }
}

/** In-place Fisher–Yates shuffle. */
function shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

/** Split a flat cell list into per-team chunks matching `sizes`. */
function splitCells<T>(flat: T[], sizes: number[]): T[][] {
    const out: T[][] = [];
    let i = 0;
    for (const size of sizes) {
        out.push(flat.slice(i, i + size));
        i += size;
    }
    return out;
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
