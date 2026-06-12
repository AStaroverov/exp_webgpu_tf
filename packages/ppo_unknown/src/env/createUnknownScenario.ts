/**
 * createUnknownScenario — builds one headless training world for ppo_unknown.
 *
 * The analogue of tanks' `createScenarioGridBase` + `createScenarioCore`: builds the
 * world a `ScenarioConfig` describes — team sizes plus how the enemy team (team 1)
 * behaves (see `scenarioCompositions`). Team 0 is always the learning policy. Steps:
 *   1. createGame headless (no render target).
 *   2. spawn `allies` + `enemies` units on distinct passable cells, each a random
 *      class (Light/Medium/Heavy — class is read from the units vector).
 *   3. drive team 0 (and team 1 under self-play) with a learning UnknownAgent;
 *      drive standing/moving enemies with a scripted RandomBot (both fire
 *      sporadically, standing just doesn't move); drive frozen enemies with a
 *      FrozenAgent (historical policy snapshot, no learning).
 *   4. install the policy driver as the SystemGroup.Before plugin (in place of the
 *      stand-in driver, which the base createGame no longer adds).
 *
 * Returns a Scenario the EpisodeManager drives: gameTick + termination/metrics.
 */

import { query, QueryResult } from "bitecs";
import { randomRangeInt } from "../../../../lib/random.ts";
import { createGame } from "../../../unknown/src/Game/createGame.ts";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { PluginDI } from "../../../unknown/src/Game/DI/PluginDI.ts";
import { SystemGroup } from "../../../unknown/src/Game/ECS/Plugins/systems.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { createTank } from "../../../unknown/src/Game/ECS/Entities/Tank/createTank.ts";
import { spawnObstacles } from "../../../unknown/src/Game/ECS/Entities/Obstacle/spawnObstacles.ts";
import { pickSpawnCells } from "../../../unknown/src/Game/Map/pickSpawnCells.ts";
import { VehicleType } from "../../../unknown/src/Game/Config/index.ts";
import {
  getTankHealth,
  getTankTeamId,
} from "../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { getTeamsCount } from "../../../unknown/src/Game/ECS/Components/TeamRef.ts";
import { ensureUnknownInputBoard } from "../state/board.ts";
import { scoreTracker } from "../reward/ScoreTracker.ts";
import { getShapingWeight } from "../reward/calculateReward.ts";
import { ScenarioConfig } from "../curriculum/types.ts";
import { UnknownAgent } from "./UnknownAgent.ts";
import { FrozenAgent } from "./FrozenAgent.ts";
import { RandomBot } from "./RandomBot.ts";
import { createPolicyDriverSystem, TankDriver } from "./createPolicyDriverSystem.ts";

const FIELD_SIZE = 1000;

// Every spawn rolls a random tank class (tanks_ml convention): type differences
// (speed, turret, reload, size) come for free from the per-type configs.
const TANK_TYPES = [
  VehicleType.LightTank,
  VehicleType.MediumTank,
  VehicleType.RocketTank,
  VehicleType.FlameTank,
  VehicleType.FrostTank,
  VehicleType.EmpTank,
] as const;

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
  world: (typeof GameDI)["world"];
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

export function createUnknownScenario(options: {
  index: number;
  train?: boolean;
  /** Team sizes and enemy behaviour — one entry of `scenarioCompositions`. */
  config: ScenarioConfig;
  /** Network iteration, drives the dense-shaping anneal (see `getShapingWeight`). */
  iteration?: number;
}): Scenario {
  const train = options.train ?? true;
  const shapingWeight = getShapingWeight(options.iteration ?? 0);
  const { allies, enemies, enemy } = options.config;
  scoreTracker.reset(); // fresh combat score per episode
  const game = createGame({ width: FIELD_SIZE, height: FIELD_SIZE });
  const world = game.world;
  const { Tank, Vehicle, VehicleController } = getGameComponents(world);
  const UnknownInputBoard = ensureUnknownInputBoard(world);

  spawnObstacles();

  const teamSizes = [allies, enemies];
  const borderSpawn = enemy === "frozen" || enemy === "self-play";
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

      const spawn = {
        playerId,
        teamId: team,
        x: pos.x,
        y: pos.y,
        color: new Float32Array(TEAM_COLORS[team % TEAM_COLORS.length]),
        rotation: Math.random() * Math.PI * 2,
      };
      const tankEid = createTank({
        ...spawn,
        type: TANK_TYPES[randomRangeInt(0, TANK_TYPES.length - 1)],
      });
      VehicleController.setMove$(tankEid, 0);
      VehicleController.setRotate$(tankEid, 0);

      const isEnemy = team !== 0;
      if (isEnemy && enemy === "standing") {
        // Holds position but occasionally fires.
        driverMap.set(tankEid, new RandomBot(tankEid, STANDING_BOT));
        continue;
      }
      if (isEnemy && enemy === "moving") {
        driverMap.set(tankEid, new RandomBot(tankEid, MOVING_BOT));
        continue;
      }
      if (isEnemy && enemy === "frozen") {
        UnknownInputBoard.addComponent(world, tankEid);
        driverMap.set(tankEid, new FrozenAgent(tankEid));
        continue;
      }

      UnknownInputBoard.addComponent(world, tankEid);
      const agent = new UnknownAgent(tankEid, train, shapingWeight);
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
  };

  function pickDistinctCells(count: number): Array<{ q: number; r: number }> {
    const grid = MapDI.grid;
    const all: Array<{ q: number; r: number }> = [];
    grid.forEachCell((cell) => {
      if (grid.isPassable(cell.q, cell.r)) all.push({ q: cell.q, r: cell.r });
    });
    shuffle(all);
    return pickSpawnCells(grid, all, count);
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

    let minX = Infinity,
      maxX = -Infinity;
    for (const c of passable) {
      if (c.x < minX) minX = c.x;
      if (c.x > maxX) maxX = c.x;
    }
    const band = (maxX - minX) * 0.25;
    const byX = [...passable].sort((a, b) => a.x - b.x); // left → right

    const takeSide = (
      count: number,
      fromLeft: boolean,
      picked: Array<{ q: number; r: number }>,
    ) => {
      // Spread randomly inside the border band; if the band can't fit the team
      // (spawn rules reject some cells), spill over to the next-nearest cells.
      const ordered = fromLeft ? byX : [...byX].reverse();
      const inBand = ordered.filter((c) => (fromLeft ? c.x <= minX + band : c.x >= maxX - band));
      const rest = ordered.filter((c) => (fromLeft ? c.x > minX + band : c.x < maxX - band));
      return pickSpawnCells(grid, [...shuffle(inBand), ...rest], count, picked);
    };

    // Team 0 → left, every other team → right; `picked` accumulates across teams
    // so no two spawn cells end up adjacent, even across the team split.
    const picked: Array<{ q: number; r: number }> = [];
    return sizes.map((count, team) => {
      const cells = takeSide(count, team === 0, picked);
      picked.push(...cells);
      return cells;
    });
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
function getTeamHealth(world: (typeof GameDI)["world"]): Record<number, number> {
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
