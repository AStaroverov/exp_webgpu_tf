/**
 * HunterBot — a scripted, non-random driver: a greedy chaser that always pressures
 * the nearest enemy. Same decision seam as `RandomBot` / `UnknownAgent` (the policy
 * driver calls `decide()` when the tank `needsDecision`), but the choice is fully
 * deterministic given the world state — no `Math.random()`.
 *
 * Each decision, in priority order:
 *   1. FIRE if it can — at the nearest allowed enemy cell (the validity mask already
 *      restricts the fire slice to enemy cells within gun reach, so "nearest allowed"
 *      is the closest shootable enemy).
 *   2. otherwise ADVANCE — step into the passable neighbour that most reduces hex
 *      distance to the nearest enemy unit (closing range so it can fire next time).
 *   3. otherwise HOLD — no shot, no enemy, or boxed in.
 *
 * Like `RandomBot` it reads `computeActionMask` purely as the source of truth for
 * legal actions (mask `0` = allowed) and hands a plain action vector to
 * `applyActionToGame` — the exact shape `batchAct` would have produced. It is a step
 * up from `RandomBot`'s undirected fire: a deterministic opponent that aims and closes.
 */

import { query } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { findPath } from "../../../unknown/src/Game/Map/findPath.ts";
import { hexDeltaDistance } from "../state/board.ts";
import {
  FIRE_ACTION_OFFSET,
  FIRE_CELL_OFFSETS,
  FIRE_TARGET_COUNT,
  HOLD_ACTION,
  MOVE_ACTION_OFFSET,
  MOVE_DIR_COUNT,
} from "../consts.ts";
import { applyActionToGame, moveDestination } from "./applyActionToGame.ts";
import { computeActionMask } from "./computeActionMask.ts";

export class HunterBot {
  constructor(
    public readonly tankEid: number,
    private readonly di = GameDI,
  ) {}

  decide(): void {
    const mask = computeActionMask(this.tankEid, this.di);

    const actions = new Float32Array(1);
    actions[0] = this.pickAction(mask);

    applyActionToGame(this.tankEid, actions, this.di);
  }

  private pickAction(mask: Float32Array): number {
    // 1. Fire at the nearest shootable enemy cell, if any fire action is allowed.
    const fire = this.nearestAllowedFire(mask);
    if (fire !== -1) return fire;

    // 2. Otherwise close in on the nearest enemy.
    const move = this.advanceTowardEnemy(mask);
    if (move !== -1) return move;

    // 3. Nothing useful to do.
    return HOLD_ACTION;
  }

  /** The allowed fire action whose target cell is closest, or -1 if none allowed. */
  private nearestAllowedFire(mask: Float32Array): number {
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < FIRE_TARGET_COUNT; i++) {
      if (mask[FIRE_ACTION_OFFSET + i] !== 0) continue; // forbidden
      const [dq, dr] = FIRE_CELL_OFFSETS[i];
      const d = hexDeltaDistance(dq, dr);
      if (d < bestDist) {
        bestDist = d;
        best = FIRE_ACTION_OFFSET + i;
      }
    }
    return best;
  }

  /**
   * The allowed move action that takes the FIRST step of the shortest A* path to
   * the nearest enemy, or -1 if no move helps (no enemy, no route, or boxed in).
   * Pathfinding (not greedy distance reduction) is what lets the bot walk AROUND
   * an obstacle between it and the enemy instead of stalling against it.
   */
  private advanceTowardEnemy(mask: Float32Array): number {
    const grid = MapDI.grid;
    if (!grid) return -1;

    const { RigidBodyState } = getGameComponents(this.di.world);
    const px = RigidBodyState.position.get(this.tankEid, 0);
    const py = RigidBodyState.position.get(this.tankEid, 1);
    const here = grid.worldToHex(px, py);
    if (!here) return -1;

    const enemy = this.nearestEnemy(here);
    if (!enemy) return -1;

    // A* around obstacles. The enemy occupies its goal cell AND reserves a full
    // buffer ring around it — so every neighbour of the goal is blocked, and a goal
    // cell allowed alone is unreachable (no neighbour to step in from). Treat the
    // target's WHOLE footprint (its cell + its reserved ring, all owned by the enemy
    // eid) as walkable for the search; any OTHER cell uses the unit's own passability
    // (its own reserved ring counts as enterable, like `moveDestination`).
    const path = findPath(grid, here, enemy, {
      isBlocked: (q, r) => {
        const cell = grid.getCell(q, r);
        if (!cell) return true;
        if (cell.occupantEid === enemy.eid) return false; // target unit + its buffer ring
        return !grid.isPassableFor(q, r, this.tankEid);
      },
    });
    if (!path || path.length < 2) return -1; // no route, or already on the goal

    // Translate the first path step (a neighbour of `here`) into its move action.
    // If that step is the enemy's own (impassable) cell, no move slot matches it →
    // -1, and we hold; we're adjacent, so the fire branch already had its chance.
    const next = path[1];
    for (let dir = 0; dir < MOVE_DIR_COUNT; dir++) {
      if (mask[MOVE_ACTION_OFFSET + dir] !== 0) continue; // not passable
      const dest = moveDestination(grid, this.tankEid, here.q, here.r, dir);
      if (dest && dest.q === next.q && dest.r === next.r) return MOVE_ACTION_OFFSET + dir;
    }
    return -1;
  }

  /** Closest enemy-team unit by hex distance — its cell and eid — or undefined. */
  private nearestEnemy(
    here: { q: number; r: number },
  ): { q: number; r: number; eid: number } | undefined {
    const grid = MapDI.grid;
    const { RigidBodyState, Tank, TeamRef, Vehicle } = getGameComponents(this.di.world);
    const myTeamId = TeamRef.id.get(this.tankEid);

    const units = query(this.di.world, [Vehicle, Tank]);
    let best: { q: number; r: number; eid: number } | undefined;
    let bestDist = Infinity;
    for (let i = 0; i < units.length; i++) {
      const eid = units[i];
      if (eid === this.tankEid) continue;
      if (TeamRef.id.get(eid) === myTeamId) continue; // ally
      const ex = RigidBodyState.position.get(eid, 0);
      const ey = RigidBodyState.position.get(eid, 1);
      const cell = grid.worldToHex(ex, ey);
      if (!cell) continue;
      const d = grid.distance(here, cell);
      if (d < bestDist) {
        bestDist = d;
        best = { q: cell.q, r: cell.r, eid };
      }
    }
    return best;
  }
}
