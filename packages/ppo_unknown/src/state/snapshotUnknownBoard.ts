/**
 * snapshotUnknownBoard — the unknown-game analogue of tanks' `snapshotTankInputTensor`.
 *
 * Fills every learning tank's `UnknownInputBoard` store with an EGOCENTRIC, POV-relative
 * view of the world — a single (2R+1)×(2R+1) window of axial deltas centered on the
 * observer (see board.ts). Occupancy is read straight off `MapDI.grid` (kept in sync by
 * `createGridOccupancySystem`), so the spatial planes touch NO physics — the board is the
 * position, the planes are the pieces. `CoordX/CoordY` are pure window geometry.
 *
 * Per observer, per window cell (dq, dr):
 *   - Off-map OR beyond VIEW_RADIUS → `Obstacle` (not enterable / not visible).
 *   - Static obstacle              → `Obstacle` plane.
 *   - The observer's own cell       → `Self` plane (always the center) + hp + stats.
 *   - Same-team unit cells          → `Ally`  plane + hp + stats.
 *   - Other-team unit cells         → `Enemy` plane + hp + stats.
 *   - `Reserved` cells              → `Reserved` plane (buffer ring around units/obstacles).
 *   - Live enemy bullet paths       → `UnderFire` plane (see `markBulletThreat`);
 *     bullets in the air are always visible regardless of who fired them.
 *   - `CoordX/CoordY`               → normalized 0..1 cell-center window coords, written
 *     for EVERY in-view cell (pure geometry, not gated by occupancy).
 *   - `EnemyHeat`: max over ALL enemies of `1 − hexDist/MAX_MAP_DIST` — the gradient
 *     that lets the agent sense enemies beyond the view radius.
 *   - `Type*` one-hot + `Reload`: the unit's `VehicleType` plane and its gun's
 *     remaining reload (log-squashed), written on the unit's own cell
 *     (self/ally/enemy).
 *
 * Enemy positions are always HONEST (real current hex) and fully observable — every
 * unit inside the view window is shown on its real cell.
 *
 * Prereq: each observing tank must have `UnknownInputBoard` added (agent setup calls
 * `UnknownInputBoard.addComponent(world, tankEid)`).
 */

import { hasComponent, query } from "bitecs";
import { GameDI } from "../../../unknown/src/Game/DI/GameDI.ts";
import { MapDI } from "../../../unknown/src/Game/DI/MapDI.ts";
import { getGameComponents } from "../../../unknown/src/Game/ECS/createGameWorld.ts";
import { HexGridConfig } from "../../../unknown/src/Game/Map/HexConfig.ts";
import { OccupantKind, type HexGrid } from "../../../unknown/src/Game/Map/HexGrid.ts";
import { getTankHealth } from "../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts";
import {
  getSlotFillerEid,
  isSlot,
  isSlotEmpty,
} from "../../../unknown/src/Game/ECS/Utils/SlotUtils.ts";
import {
  BOARD_COLS,
  BOARD_ROWS,
  BoardChannel,
  ensureUnknownInputBoard,
  hexDeltaDistance,
  VIEW_RADIUS,
  type UnknownInputBoardComponent,
} from "./board.ts";
import { VehicleType } from "../../../unknown/src/Game/Config/vehicles.ts";
import { markBulletThreat } from "./markBulletThreat.ts";
import { needsDecision } from "../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts";

type GameComponents = ReturnType<typeof getGameComponents>;

/** Enemies in the world this tick: parallel arrays of real hex + heat weight (1). */
type KnownEnemies = { q: number[]; r: number[]; w: number[] };

/** Everything the per-cell writers need — bundled so each step takes a tiny signature. */
type SnapshotCtx = {
  grid: HexGrid;
  selfEid: number;
  selfQ: number;
  selfR: number;
  myTeamId: number;
  enemies: KnownEnemies;
  world: (typeof GameDI)["world"];
  UnknownInputBoard: UnknownInputBoardComponent;
};

/** `VehicleType` → its one-hot board plane (non-tank types have no plane). */
const TANK_TYPE_CHANNEL: Partial<Record<VehicleType, number>> = {
  [VehicleType.LightTank]: BoardChannel.TypeLightTank,
  [VehicleType.MediumTank]: BoardChannel.TypeMediumTank,
  [VehicleType.RocketTank]: BoardChannel.TypeRocketTank,
  [VehicleType.FlameTank]: BoardChannel.TypeFlameTank,
  [VehicleType.FrostTank]: BoardChannel.TypeFrostTank,
  [VehicleType.EmpTank]: BoardChannel.TypeEmpTank,
};

export function snapshotUnknownBoard({ world } = GameDI) {
  const grid = MapDI.grid;
  if (!grid) return;

  const { Tank, Vehicle, TeamRef } = getGameComponents(world);
  const UnknownInputBoard = ensureUnknownInputBoard(world);
  const observers = query(world, [Vehicle, Tank, UnknownInputBoard]);

  for (let i = 0; i < observers.length; i++) {
    const selfEid = observers[i];
    if (!needsDecision(selfEid)) continue;

    const myTeamId = TeamRef.id.get(selfEid);
    const field = scanField(grid, selfEid, myTeamId, TeamRef);
    if (!field) continue; // not on the grid (mid-transition) — keep last snapshot

    UnknownInputBoard.reset(selfEid);

    const ctx: SnapshotCtx = {
      grid,
      selfEid,
      selfQ: field.selfQ,
      selfR: field.selfR,
      myTeamId,
      enemies: field.enemies,
      world,
      UnknownInputBoard,
    };
    fillWindow(ctx);

    // Threat from live enemy bullets — straight-line projection of each
    // bullet's remaining (fixed-distance) flight path onto the window.
    markBulletThreat(selfEid, myTeamId, field.selfQ, field.selfR, grid, world);
  }
}

/**
 * Sweep the grid once for this observer: locate its own hex and every enemy, at the
 * enemies' real current hexes. Returns null when the observer isn't on the grid
 * (mid-transition).
 */
function scanField(
  grid: HexGrid,
  selfEid: number,
  myTeam: number,
  TeamRef: GameComponents["TeamRef"],
): { selfQ: number; selfR: number; enemies: KnownEnemies } | null {
  let selfQ = NaN;
  let selfR = NaN;
  const enemies: KnownEnemies = { q: [], r: [], w: [] };
  grid.forEachCell((cell, hex) => {
    if (cell.occupantKind !== OccupantKind.Unit) return;
    const unitEid = cell.occupantEid!;
    if (unitEid === selfEid) {
      selfQ = hex.q;
      selfR = hex.r;
    } else if (TeamRef.id.get(unitEid) !== myTeam) {
      enemies.q.push(hex.q);
      enemies.r.push(hex.r);
      enemies.w.push(1);
    }
  });
  if (Number.isNaN(selfQ)) return null;
  return { selfQ, selfR, enemies };
}

/** Pass 2: fill the egocentric (2R+1)² window, one cell at a time. */
function fillWindow(ctx: SnapshotCtx) {
  for (let dr = -VIEW_RADIUS; dr <= VIEW_RADIUS; dr++) {
    for (let dq = -VIEW_RADIUS; dq <= VIEW_RADIUS; dq++) {
      fillCell(ctx, dq, dr);
    }
  }
}

function fillCell(ctx: SnapshotCtx, dq: number, dr: number) {
  writeEnemyHeat(ctx, dq, dr);

  // Beyond the view radius (the window square's corners) — not visible.
  if (hexDeltaDistance(dq, dr) > VIEW_RADIUS) {
    markObstacle(ctx, dq, dr);
    return;
  }

  writeWindowGeometry(ctx, dq, dr);
  writeOccupant(ctx, dq, dr);
}

function writeEnemyHeat(ctx: SnapshotCtx, dq: number, dr: number) {
  const { enemies, selfQ, selfR, UnknownInputBoard } = ctx;
  let heat = 0;
  for (let e = 0; e < enemies.q.length; e++) {
    const d = hexDeltaDistance(selfQ + dq - enemies.q[e], selfR + dr - enemies.r[e]);
    heat = Math.max(
      heat,
      enemies.w[e] * (1 - d / Math.hypot(HexGridConfig.cols, HexGridConfig.rows)),
    );
  }
  if (heat > 0) UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.EnemyHeat, heat);
}

/** CoordX/CoordY: normalized col/row over the window (0..1), pure geometry. */
function writeWindowGeometry(ctx: SnapshotCtx, dq: number, dr: number) {
  const col = dq + VIEW_RADIUS;
  const row = dr + VIEW_RADIUS;
  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.CoordX, col / (BOARD_COLS - 1));
  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.CoordY, row / (BOARD_ROWS - 1));
}

/** Dispatch an in-view cell's occupancy to the matching plane. */
function writeOccupant(ctx: SnapshotCtx, dq: number, dr: number) {
  const cell = ctx.grid.getCell(ctx.selfQ + dq, ctx.selfR + dr);
  if (!cell) {
    markObstacle(ctx, dq, dr); // off-map — not enterable, not visible
    return;
  }
  switch (cell.occupantKind) {
    case OccupantKind.Obstacle:
      markObstacle(ctx, dq, dr);
      return;
    case OccupantKind.Reserved:
      writeReserved(ctx, dq, dr, cell.occupantEid!);
      return;
    case OccupantKind.Unit:
      writeUnit(ctx, dq, dr, cell.occupantEid!);
      return;
    default:
      return; // empty (null)
  }
}

function markObstacle(ctx: SnapshotCtx, dq: number, dr: number) {
  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.Obstacle, 1);
}

/** Reserved cell (buffer ring around a unit or obstacle). */
function writeReserved(ctx: SnapshotCtx, dq: number, dr: number, _reserverEid: number) {
  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.Reserved, 1);
}

/** Unit cell → Self/Ally/Enemy plane + hp + stats. */
function writeUnit(ctx: SnapshotCtx, dq: number, dr: number, unitEid: number) {
  const { TeamRef } = getGameComponents(ctx.world);
  const plane =
    unitEid === ctx.selfEid
      ? BoardChannel.Self
      : TeamRef.id.get(unitEid) === ctx.myTeamId
        ? BoardChannel.Ally
        : BoardChannel.Enemy;

  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, plane, 1);
  ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.Hp, getTankHealth(unitEid));
  writeReload(ctx, dq, dr, unitEid);
  writeDamageStatuses(ctx, dq, dr, unitEid);
  writeTypeOneHot(ctx, dq, dr, unitEid);
}

/**
 * Damage-status channels of the unit on cell (dq, dr): per-part `Dot` summed up
 * to the vehicle and log-squashed (`log1p(sum / flame_dps)` — 0 when clean, ~0.7
 * at one flame stack, grows slowly with more), and the vehicle-level slow
 * multiplier (dense default `1` = not slowed, full speed).
 */
function writeDamageStatuses(ctx: SnapshotCtx, dq: number, dr: number, unitEid: number) {
  const { selfEid, world, UnknownInputBoard } = ctx;
  const { Tank, Slowed } = getGameComponents(world);

  // The same part set getTankHealth counts: hull slots + turret slots.
  const dps = sumDotDps(ctx, unitEid) + sumDotDps(ctx, Tank.turretEId.get(unitEid));

  if (dps > 0) {
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Dot, Math.log1p(dps));
  }

  if (hasComponent(world, unitEid, Slowed)) {
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Slow, Slowed.slowMul.get(unitEid));
  }
}

/** Summed dot dps over one parent's filled slots. */
function sumDotDps(ctx: SnapshotCtx, parentEid: number): number {
  const { world } = ctx;
  const { Children, Dot } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEid);
  let dps = 0;
  for (let i = 0; i < childCount; i++) {
    const slotEid = Children.entitiesIds.get(parentEid, i);
    if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;
    const partEid = getSlotFillerEid(slotEid);
    if (partEid === 0) continue;
    dps += Dot.dps.get(partEid); // a part without a Dot reads 0
  }
  return dps;
}

/** One-hot `VehicleType` plane of the unit on cell (dq, dr). */
function writeTypeOneHot(ctx: SnapshotCtx, dq: number, dr: number, unitEid: number) {
  const { Vehicle } = getGameComponents(ctx.world);
  const channel = TANK_TYPE_CHANNEL[Vehicle.type.get(unitEid) as VehicleType];
  if (channel !== undefined) {
    ctx.UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, channel, 1);
  }
}

/**
 * How "unready to fire" the unit's gun is. 0 (the dense default) = ready.
 * Bullet guns: remaining reload, log-squashed `log1p(remaining_ms / 1000)`.
 * Stream guns: the charge deficit `1 - charge` (0 = full, ~1 = empty) — already
 * a dense 0..1 signal of comparable scale, no squashing needed. Gunless vehicles stay 0.
 */
function writeReload(ctx: SnapshotCtx, dq: number, dr: number, unitEid: number) {
  const { selfEid, world, UnknownInputBoard } = ctx;
  const { Tank, Firearms, StreamFirearms } = getGameComponents(world);
  const turretEid = Tank.turretEId.get(unitEid);
  if (turretEid === 0) return;
  let value = 0;
  if (hasComponent(world, turretEid, Firearms)) {
    value = Math.log1p(Firearms.reloading.get(turretEid) / 1000);
  } else if (hasComponent(world, turretEid, StreamFirearms)) {
    value = 1 - StreamFirearms.getCharge(turretEid);
  }
  if (value > 0) {
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Reload, value);
  }
}
