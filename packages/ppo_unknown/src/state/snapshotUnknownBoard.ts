/**
 * snapshotUnknownBoard — the unknown-game analogue of tanks' `snapshotTankInputTensor`.
 *
 * Fills every learning tank's `UnknownInputBoard` store with an EGOCENTRIC, POV-relative
 * view of the world — a single (2R+1)×(2R+1) window of axial deltas centered on the
 * observer (see board.ts). Occupancy is read straight off `MapDI.grid` (kept in sync by
 * `createGridOccupancySystem`), so the spatial planes touch NO physics — the board is the
 * position, the planes are the pieces. The light channel `UnderBeam` comes from the
 * spotting system's beam-cell accessor; `CoordX/CoordY` are pure window geometry.
 *
 * Per observer, per window cell (dq, dr):
 *   - Off-map OR beyond VIEW_RADIUS → `Obstacle` (not enterable / not visible).
 *   - Static obstacle              → `Obstacle` plane.
 *   - The observer's own cell       → `Self` plane (always the center) + hp + stats +
 *     `SpotConfidence` = "am I spotted by the enemy" (`getConfidence` — single value).
 *   - Same-team unit cells          → `Ally`  plane + hp + stats.
 *   - Other-team unit cells         → `Enemy` plane + hp + stats + `SpotConfidence`, but
 *     ONLY when the enemy is currently visible to the observer's team (Spotting).
 *   - `Reserved` cells              → `Reserved` plane (a unit is driving into them),
 *     but ONLY when the reserving unit is an ally or a visible enemy — an unspotted
 *     enemy's reservation must not leak.
 *   - Live enemy bullet paths       → `UnderFire` plane (see `markBulletThreat`);
 *     bullets in the air are always visible regardless of who fired them.
 *   - `CoordX/CoordY`               → normalized 0..1 cell-center window coords, written
 *     for EVERY in-view cell (pure geometry, not gated by occupancy).
 *   - `UnderBeam`                   → every in-view hex covered by any Ranger searchlight
 *     beam this tick (`isBeamCell` from the spotting system); always visible, like live
 *     bullets, regardless of whether the emitting Ranger is itself spotted.
 *   - `EnemyHeat`: max over enemies KNOWN to the observer's team of
 *     `confidence · (1 − hexDist/MAX_MAP_DIST)` — the gradient that lets the agent sense
 *     enemies beyond the view radius.
 *   - `Role/Mobility/Firepower/Reload/Range`: the unit's normalized identity + combat
 *     stats from `vehicleStats.ts`, written on the unit's own cell (self/ally/enemy).
 *
 * Spotting (засвет) rules: enemy positions are always HONEST (real current hex); each
 * enemy contributes only if the opposing side has spotted him recently. Confidence is a
 * single per-victim value (two-sided game), so `Spottable.getConfidence(enemyEid)` is
 * the fading weight (1 right after a spot, → 0 over a 3 s memory window) that scales his
 * heat AND fills `SpotConfidence`; `Spottable.isVisible` gates the discrete Enemy/Hp/stat
 * planes (proximity / searchlight this very tick). An enemy revealed only by firing has
 * heat but is not "visible".
 *
 * Prereq: each observing tank must have `UnknownInputBoard` added (agent setup calls
 * `UnknownInputBoard.addComponent(world, tankEid)`).
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { HexGridConfig } from '../../../unknown/src/Game/Map/HexConfig.ts';
import { OccupantKind, type HexGrid } from '../../../unknown/src/Game/Map/HexGrid.ts';
import { getTankHealth } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { isBeamCell } from '../../../unknown/src/Game/ECS/Systems/Spotting/createSpottingSystem.ts';
import {
    BOARD_COLS,
    BOARD_ROWS,
    BoardChannel,
    hexDeltaDistance,
    UnknownInputBoard,
    VIEW_RADIUS,
} from './board.ts';
import { getVehicleStats } from './vehicleStats.ts';
import { markBulletThreat } from './markBulletThreat.ts';
import { needsDecision } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';


type GameComponents = ReturnType<typeof getGameComponents>;

/** Enemies KNOWN to the observer's team this tick: parallel arrays of real hex + fading weight. */
type KnownEnemies = { q: number[]; r: number[]; w: number[] };

/** Everything the per-cell writers need — bundled so each step takes a tiny signature. */
type SnapshotCtx = {
    selfEid: number;
    selfQ: number;
    selfR: number;
    myTeamId: number;
    grid: HexGrid;
    enemies: KnownEnemies;
    Vehicle: GameComponents['Vehicle'];
    TeamRef: GameComponents['TeamRef'];
    Spottable: GameComponents['Spottable'];
};

export function snapshotUnknownBoard({ world } = GameDI) {
    const grid = MapDI.grid;
    if (!grid) return;

    const { Tank, Vehicle, TeamRef, Spottable } = getGameComponents(world);
    const observers = query(world, [Vehicle, Tank, UnknownInputBoard]);

    for (let i = 0; i < observers.length; i++) {
        const selfEid = observers[i];
        if (!needsDecision(selfEid)) continue;

        const myTeamId = TeamRef.id[selfEid];
        const field = scanField(grid, selfEid, myTeamId, TeamRef, Spottable);
        if (!field) continue; // not on the grid (mid-transition) — keep last snapshot

        UnknownInputBoard.reset(selfEid);

        const ctx: SnapshotCtx = {
            grid,
            selfEid,
            selfQ: field.selfQ,
            selfR: field.selfR,
            myTeamId,
            enemies: field.enemies,
            Vehicle,
            TeamRef,
            Spottable,
        };
        fillWindow(ctx);

        // Threat from live enemy bullets — straight-line projection of each
        // bullet's remaining (fixed-distance) flight path onto the window.
        markBulletThreat(selfEid, myTeamId, field.selfQ, field.selfR, grid, world);
    }
}

/**
 * Sweep the grid once for this observer: locate its own hex and every enemy KNOWN to
 * its team (confidence > 0), at the enemies' real current hexes. Returns null when the
 * observer isn't on the grid (mid-transition).
 */
function scanField(
    grid: HexGrid,
    selfEid: number,
    myTeam: number,
    TeamRef: GameComponents['TeamRef'],
    Spottable: GameComponents['Spottable'],
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
        } else if (TeamRef.id[unitEid] !== myTeam) {
            const w = Spottable.getConfidence(unitEid);
            if (w === 0) return; // unknown to my team — does not exist for this observer
            enemies.q.push(hex.q);
            enemies.r.push(hex.r);
            enemies.w.push(w);
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
    writeUnderBeam(ctx, dq, dr);
    writeOccupant(ctx, dq, dr);
}


function writeEnemyHeat(ctx: SnapshotCtx, dq: number, dr: number) {
    const { enemies, selfQ, selfR } = ctx;
    let heat = 0;
    for (let e = 0; e < enemies.q.length; e++) {
        const d = hexDeltaDistance(selfQ + dq - enemies.q[e], selfR + dr - enemies.r[e]);
        heat = Math.max(heat, enemies.w[e] * (1 - d / Math.hypot(HexGridConfig.cols, HexGridConfig.rows)));
    }
    if (heat > 0) UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.EnemyHeat, heat);
}

/** CoordX/CoordY: normalized col/row over the window (0..1), pure geometry. */
function writeWindowGeometry(ctx: SnapshotCtx, dq: number, dr: number) {
    const col = dq + VIEW_RADIUS;
    const row = dr + VIEW_RADIUS;
    UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.CoordX, col / (BOARD_COLS - 1));
    UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.CoordY, row / (BOARD_ROWS - 1));
}

/** UnderBeam: any hex lit by a Ranger searchlight this tick. */
function writeUnderBeam(ctx: SnapshotCtx, dq: number, dr: number) {
    if (isBeamCell(ctx.selfQ + dq, ctx.selfR + dr)) {
        UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.UnderBeam, 1);
    }
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
    UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.Obstacle, 1);
}

/**
 * Reserved cell (a unit is driving into it). Don't leak an unspotted enemy's
 * reservation: mark only when the reserving unit is an ally or a visible enemy.
 */
function writeReserved(ctx: SnapshotCtx, dq: number, dr: number, reserverEid: number) {
    const reserverIsAlly = ctx.TeamRef.id[reserverEid] === ctx.myTeamId;
    if (reserverIsAlly || ctx.Spottable.isVisible(reserverEid)) {
        UnknownInputBoard.setDelta(ctx.selfEid, dq, dr, BoardChannel.Reserved, 1);
    }
}

/** Unit cell → Self/Ally/Enemy plane + hp + stats + SpotConfidence. */
function writeUnit(ctx: SnapshotCtx, dq: number, dr: number, unitEid: number) {
    const { selfEid, myTeamId: myTeam, Spottable } = ctx;
    const isSelf = unitEid === selfEid;
    const isAlly = ctx.TeamRef.id[unitEid] === myTeam;

    // An enemy occupies the cell only when currently spotted; otherwise the cell
    // reads as empty (he still feeds heat above).
    if (!isSelf && !isAlly && !Spottable.isVisible(unitEid)) return;

    const plane = isSelf ? BoardChannel.Self : isAlly ? BoardChannel.Ally : BoardChannel.Enemy;

    UnknownInputBoard.setDelta(selfEid, dq, dr, plane, 1);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Hp, getTankHealth(unitEid));
    writeStats(selfEid, dq, dr, unitEid, ctx.Vehicle);

    // SpotConfidence (single per-victim value): on an enemy cell, how confidently I see
    // him; on my own cell, how confidently the enemy sees me ("am I spotted").
    if (isSelf || plane === BoardChannel.Enemy) {
        UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.SpotConfidence,
            Spottable.getConfidence(unitEid));
    }
}

/**
 * Write the normalized vehicle-stat channels (role/mobility/firepower/reload/range) for
 * the unit on cell (dq, dr) into the observer's board. `unitEid` is the unit whose
 * vehicle type the stats come from.
 */
function writeStats(
    selfEid: number,
    dq: number,
    dr: number,
    unitEid: number,
    Vehicle: { type: ArrayLike<number> },
) {
    const stats = getVehicleStats(Vehicle.type[unitEid]);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Role, stats.role);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Mobility, stats.mobility);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Firepower, stats.firepower);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Reload, stats.reload);
    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Range, stats.range);
}
