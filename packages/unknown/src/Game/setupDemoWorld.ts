/**
 * setupDemoWorld — build-specific world content for the dev/demo game: a 1v1
 * duel. Two medium tanks spawn facing each other across the field center —
 * team 1 (west) is the human's tank (mouse + keyboard, marked `PlayerControlled`
 * so no AI driver touches it), team 2 (east) is the opponent. This builder spawns
 * the units only; the opponent's decision driver is attached by the caller
 * (`index.ts` wires the trained policy via `attachPolicyOpponent`). Returns both
 * tank eids.
 *
 * Extracted out of `createGame` so the base game wires only systems. Training
 * builds (`ppo_unknown`) skip this entirely and spawn their own teams + policy
 * driver via `createUnknownScenario`.
 *
 * Must be called AFTER `createGame()` (it relies on the live `GameDI`/`MapDI`).
 */

import { query } from "bitecs";
import { GameDI } from "./DI/GameDI.ts";
import { MapDI } from "./DI/MapDI.ts";
import { PluginDI } from "./DI/PluginDI.ts";
import { getGameComponents } from "./ECS/createGameWorld.ts";
import { SystemGroup } from "./ECS/Plugins/systems.ts";
import { createTank, type TankVehicleType } from "./ECS/Entities/Tank/createTank.ts";
import { VehicleType, teamBaseColor } from "./Config/index.ts";
import { spawnObstacles } from "./ECS/Entities/Obstacle/spawnObstacles.ts";
import { spawnBorderRocks } from "./ECS/Entities/Obstacle/spawnBorderRocks.ts";
import { HexGridConfig } from "./Map/HexConfig.ts";

/** Enemy classes the wave spawner rolls from (the set the policy was trained on). */
const ENEMY_TYPES: readonly TankVehicleType[] = [
  VehicleType.LightTank,
  VehicleType.MediumTank,
  VehicleType.RocketTank,
  VehicleType.FlameTank,
  VehicleType.FrostTank,
  VehicleType.EmpTank,
];

/** Most enemies (team 2) alive at once; the wave spawner skips while at the cap. */
const MAX_ENEMIES = 5;
/** Cells in from the left/right border the two starting tanks spawn at. */
const SPAWN_INSET_CELLS = 3;
/** World distance between adjacent pointy-hex columns (√3 · radius). */
const HEX_STEP_X = Math.sqrt(3) * HexGridConfig.radius;
import { createShapeCountDiagnosticSystem } from "./ECS/Plugins/createShapeCountDiagnosticSystem.ts";
import { createRepairSystem } from "./ECS/Systems/createRepairSystem.ts";

export type DemoWorld = {
  playerEid: number;
  enemyEid: number;
  /** Spawn another team-2 enemy on a random passable cell; null if the field is full. */
  spawnEnemy: () => number | null;
};

export function setupDemoWorld({ world } = GameDI): DemoWorld {
  const { VehicleController, PlayerControlled, Repairer, Score, Tank, TeamRef } =
    getGameComponents(world);
  let nextEnemyPlayerId = 3; // 1 = player, 2 = the starting enemy

  // TEMP diagnostic (DELETE once the 10k shape-buffer overflow is diagnosed).
  PluginDI.addSystem(SystemGroup.After, createShapeCountDiagnosticSystem());

  // Player-only self-repair: salvage ground scrap to refill slots (runs after
  // physics so it reads fresh positions, before destroyFrame reaps the scrap).
  PluginDI.addSystem(SystemGroup.Before, createRepairSystem());

  spawnBorderRocks(); // rocky wall around the arena (the off-screen contour)
  spawnObstacles(); // scattered interior cover (avoids the now-occupied border)

  const grid = MapDI.grid;
  const bounds = grid.worldBounds();
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;

  // Start on opposite sides, ~SPAWN_INSET_CELLS in from the left/right border and
  // centered vertically — player left, enemy right, each facing the field center.
  const playerCell = nearestPassableCell(bounds.minX + SPAWN_INSET_CELLS * HEX_STEP_X, cy);
  const enemyCell = nearestPassableCell(bounds.maxX - SPAWN_INSET_CELLS * HEX_STEP_X, cy);
  if (!playerCell || !enemyCell) throw new Error("[setupDemoWorld] no passable spawn cells");

  const pPos = grid.hexToWorld(playerCell.q, playerCell.r)!;
  const ePos = grid.hexToWorld(enemyCell.q, enemyCell.r)!;

  // Team 1 — the human's tank.
  const playerEid = createTank({
    type: VehicleType.MediumTank,
    playerId: 1,
    teamId: 1,
    x: pPos.x,
    y: pPos.y,
    rotation: Math.atan2(cy - pPos.y, cx - pPos.x),
    color: new Float32Array(teamBaseColor(1)),
  });
  PlayerControlled.addComponent(world, playerEid);
  Repairer.addComponent(world, playerEid); // can heal by salvaging ground scrap
  Score.addComponent(world, playerEid); // running score, credited by enemy damage dealt

  // Team 2 — the opponent; its driver is attached by the caller.
  const enemyEid = createTank({
    type: VehicleType.MediumTank,
    playerId: 2,
    teamId: 2,
    x: ePos.x,
    y: ePos.y,
    rotation: Math.atan2(cy - ePos.y, cx - ePos.x),
    color: new Float32Array(teamBaseColor(2)),
  });

  for (const eid of [playerEid, enemyEid]) {
    VehicleController.setMove$(eid, 0);
    VehicleController.setRotate$(eid, 0);
  }

  return { playerEid, enemyEid, spawnEnemy };

  /** Spawn one more team-2 enemy on a random passable cell (null if none free or at the cap). */
  function spawnEnemy(): number | null {
    const tanks = query(world, [Tank, TeamRef]);
    let enemyCount = 0;
    for (let i = 0; i < tanks.length; i++) {
      if (TeamRef.id.get(tanks[i]) === 2) enemyCount++;
    }
    if (enemyCount >= MAX_ENEMIES) return null;

    const cell = pickRandomPassableCell();
    if (!cell) return null;
    const pos = grid.hexToWorld(cell.q, cell.r);
    if (!pos) return null;

    const type = ENEMY_TYPES[Math.floor(Math.random() * ENEMY_TYPES.length)];
    const eid = createTank({
      type,
      playerId: nextEnemyPlayerId++,
      teamId: 2,
      x: pos.x,
      y: pos.y,
      rotation: Math.random() * Math.PI * 2,
      color: new Float32Array(teamBaseColor(2)),
    });
    VehicleController.setMove$(eid, 0);
    VehicleController.setRotate$(eid, 0);
    return eid;
  }

  /** A random currently-passable cell (no unit/obstacle/reservation), or null. */
  function pickRandomPassableCell(): { q: number; r: number } | null {
    const free: Array<{ q: number; r: number }> = [];
    grid.forEachCell((cell) => {
      if (grid.isPassable(cell.q, cell.r)) free.push({ q: cell.q, r: cell.r });
    });
    return free.length > 0 ? free[Math.floor(Math.random() * free.length)] : null;
  }

  /** The passable cell whose world center is closest to `(x, y)`, or null. */
  function nearestPassableCell(x: number, y: number): { q: number; r: number } | null {
    let best: { q: number; r: number } | null = null;
    let bestDist = Infinity;
    grid.forEachCell((cell) => {
      if (!grid.isPassable(cell.q, cell.r)) return;
      const pos = grid.hexToWorld(cell.q, cell.r);
      if (!pos) return;
      const d = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);
      if (d < bestDist) {
        bestDist = d;
        best = { q: cell.q, r: cell.r };
      }
    });
    return best;
  }
}
