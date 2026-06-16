/**
 * createBurnScenario — the single-thread analogue of ppo_unknown's
 * `createUnknownScenario`. It builds the SAME real headless `unknown` game world
 * (createGame + obstacles + tank spawns + the policy-driver plugin) and exposes the
 * SAME `Scenario` shape, but drives the learning team with `BurnUnknownAgent`
 * (V4Trainer inference) instead of the tfjs `UnknownAgent`.
 *
 * It reuses ppo_unknown's real modules unchanged:
 *   - the game itself (`createGame`, `createTank`, `spawnObstacles`, the grid, configs),
 *   - `createPolicyDriverSystem` (decision scheduling / drain),
 *   - `RandomBot` (scripted enemies — no network),
 *   - `scoreTracker` + `getShapingWeight` (reward bookkeeping),
 *   - `ensureUnknownInputBoard` (observation store).
 *
 * Only the agent class differs. The tfjs `FrozenAgent` (historical IndexedDB snapshot)
 * has no single-thread analogue, so 'frozen'/'self-play' enemies are driven by a live
 * `BurnUnknownAgent` in inference mode (train=false → acts, records nothing) sharing the
 * one in-process policy. The success-ratio / termination / metrics contract is identical.
 */

import { query, QueryResult } from "bitecs";
import { randomRangeInt } from "../../../lib/random.ts";
import { createGame } from "../../unknown/src/Game/createGame.ts";
import { GameDI } from "../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../unknown/src/Game/DI/MapDI.ts";
import { PluginDI } from "../../unknown/src/Game/DI/PluginDI.ts";
import { SystemGroup } from "../../unknown/src/Game/ECS/Plugins/systems.ts";
import { getGameComponents } from "../../unknown/src/Game/ECS/createGameWorld.ts";
import { createTank } from "../../unknown/src/Game/ECS/Entities/Tank/createTank.ts";
import { spawnObstacles } from "../../unknown/src/Game/ECS/Entities/Obstacle/spawnObstacles.ts";
import { pickSpawnCells } from "../../unknown/src/Game/Map/pickSpawnCells.ts";
import { VehicleType } from "../../unknown/src/Game/Config/index.ts";
import {
  getTankHealth,
  getTankTeamId,
} from "../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import { getTeamsCount } from "../../unknown/src/Game/ECS/Components/TeamRef.ts";
import { ensureUnknownInputBoard } from "../../ppo_unknown/src/state/board.ts";
import { scoreTracker } from "../../ppo_unknown/src/reward/ScoreTracker.ts";
import { getShapingWeight } from "../../ppo_unknown/src/reward/calculateReward.ts";
import { RandomBot } from "../../ppo_unknown/src/env/RandomBot.ts";
import {
  createPolicyDriverSystem,
  TankDriver,
} from "../../ppo_unknown/src/env/createPolicyDriverSystem.ts";
import type { ScenarioConfig } from "../../ppo_unknown/src/curriculum/types.ts";
import { BurnUnknownAgent } from "./BurnUnknownAgent.ts";

const FIELD_SIZE = 1000;

const TANK_TYPES = [
  VehicleType.LightTank,
  VehicleType.MediumTank,
  VehicleType.RocketTank,
  VehicleType.FlameTank,
  VehicleType.FrostTank,
  VehicleType.EmpTank,
] as const;

const STANDING_BOT = { moveProb: 0, fireProb: 0.1 };
const MOVING_BOT = { moveProb: 0.3, fireProb: 0.2 };

const TEAM_COLORS: Array<[number, number, number, number]> = [
  [1.0, 0.4, 0.4, 1],
  [0.4, 0.7, 1.0, 1],
];

export type BurnScenario = {
  world: (typeof GameDI)["world"];
  index: number;
  train: boolean;
  width: number;
  height: number;
  agents: BurnUnknownAgent[];
  gameTick: (delta: number) => void;
  drainDecisions: () => Promise<void>;
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
  getVehicleEids: () => QueryResult;
  getTeamsCount: () => number;
  getSuccessRatio: () => number;
};

export function createBurnScenario(options: {
  index: number;
  train?: boolean;
  config: ScenarioConfig;
  iteration?: number;
}): BurnScenario {
  const train = options.train ?? true;
  const shapingWeight = getShapingWeight(options.iteration ?? 0);
  const { allies, enemies, enemy } = options.config;
  scoreTracker.reset();
  const game = createGame({ width: FIELD_SIZE, height: FIELD_SIZE });
  const world = game.world;
  const { Tank, Vehicle, VehicleController } = getGameComponents(world);
  const UnknownInputBoard = ensureUnknownInputBoard(world);

  spawnObstacles();

  const teamSizes = [allies, enemies];
  const teamCells = splitCells(pickDistinctCells(allies + enemies), teamSizes);
  const agents: BurnUnknownAgent[] = [];
  const driverMap = new Map<number, TankDriver>();

  // 'frozen' / 'self-play' enemies share the live policy in inference mode.
  const enemyUsesPolicy = enemy === "frozen" || enemy === "self-play";

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
        driverMap.set(tankEid, new RandomBot(tankEid, STANDING_BOT));
        continue;
      }
      if (isEnemy && enemy === "moving") {
        driverMap.set(tankEid, new RandomBot(tankEid, MOVING_BOT));
        continue;
      }

      UnknownInputBoard.addComponent(world, tankEid);
      // Enemies sharing the policy never record training data (train=false).
      const isTrainer = !isEnemy && train;
      const agent = new BurnUnknownAgent(tankEid, isTrainer, shapingWeight);
      if (isTrainer) agents.push(agent);
      else if (isEnemy && !enemyUsesPolicy) {
        // Defensive: any other enemy mode falls back to a moving bot.
        driverMap.set(tankEid, new RandomBot(tankEid, MOVING_BOT));
        continue;
      }
      driverMap.set(tankEid, agent);
    }
  }

  const policyDriver = createPolicyDriverSystem(driverMap);
  PluginDI.addSystem(SystemGroup.Before, policyDriver.system);

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
    drainDecisions: policyDriver.drain,
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
}

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function splitCells<T>(flat: T[], sizes: number[]): T[][] {
  const out: T[][] = [];
  let i = 0;
  for (const size of sizes) {
    out.push(flat.slice(i, i + size));
    i += size;
  }
  return out;
}

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
