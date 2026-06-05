/**
 * snapshotUnknownBoard — the unknown-game analogue of tanks' `snapshotTankInputTensor`.
 *
 * Fills every learning tank's `UnknownInputBoard` with an EGOCENTRIC, POV-relative
 * view of the hex grid: a (2R+1)×(2R+1) window of axial deltas centered on the
 * observer (see board.ts). Occupancy is read straight off `MapDI.grid` (kept in
 * sync by `createGridOccupancySystem`), so this touches NO physics — exactly the
 * strategic principle: the board is the position, the planes are the pieces.
 *
 * Per observer, per window cell (dq, dr):
 *   - Off-map OR beyond VIEW_RADIUS → `Obstacle` (not enterable / not visible).
 *   - Static obstacle              → `Obstacle` plane.
 *   - The observer's own cell      → `Self` plane (always the center, + its hp).
 *   - Same-team unit cells         → `Ally`  plane (+ hp).
 *   - Other-team unit cells        → `Enemy` plane (+ hp).
 *   - `Reserved` cells             → `Reserved` plane (a unit is driving into them).
 *   - Live enemy bullet paths      → `UnderFire` plane (see `markBulletThreat`).
 *   - `EnemyHeat`: max over ALL enemies of `1 − hexDist/MAX_MAP_DIST` — the
 *     gradient that lets the agent sense enemies beyond the view radius.
 *
 * Prereq: each observing tank must have `UnknownInputBoard` added (agent setup calls
 * `UnknownInputBoard.addComponent(world, tankEid)`).
 */

import { query } from 'bitecs';
import { Grid, rectangle } from 'honeycomb-grid';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { HexGridConfig, HexTile } from '../../../unknown/src/Game/Map/HexConfig.ts';
import { OccupantKind } from '../../../unknown/src/Game/Map/HexGrid.ts';
import { getTankHealth } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { BoardChannel, hexDeltaDistance, UnknownInputBoard, VIEW_RADIUS } from './board.ts';
import { markBulletThreat } from './markBulletThreat.ts';
import { needsDecision } from '../../../unknown/src/Game/ECS/Actions/ActionSchedule.ts';

/**
 * Max hex distance between any two map cells — the `EnemyHeat` normalizer
 * ("1 on the enemy, 0 at the farthest possible distance"). Computed once from
 * the map rectangle's corners (a convex shape's diameter is corner-to-corner).
 */
export const MAX_MAP_DIST = (() => {
    const grid = new Grid(HexTile, rectangle({ width: HexGridConfig.cols, height: HexGridConfig.rows }));
    const lastCol = HexGridConfig.cols - 1;
    const lastRow = HexGridConfig.rows - 1;
    const corners = [
        { col: 0, row: 0 },
        { col: lastCol, row: 0 },
        { col: 0, row: lastRow },
        { col: lastCol, row: lastRow },
    ];
    let max = 0;
    for (let i = 0; i < corners.length; i++) {
        for (let j = i + 1; j < corners.length; j++) {
            max = Math.max(max, grid.distance(corners[i], corners[j]));
        }
    }
    return max;
})();

/** Scratch list of enemy axial coords, reused across observers. */
const enemyQ: number[] = [];
const enemyR: number[] = [];

export function snapshotUnknownBoard({ world } = GameDI) {
    const grid = MapDI.grid;
    if (!grid) return;

    const { Tank, Vehicle, TeamRef } = getGameComponents(world);
    const observers = query(world, [Vehicle, Tank, UnknownInputBoard]);

    for (let i = 0; i < observers.length; i++) {
        const selfEid = observers[i];
        const myTeam = TeamRef.id[selfEid];

        if (!needsDecision(selfEid)) continue;

        // Pass 1: locate self and ALL enemies (heat sources, even out of view).
        let selfQ = NaN;
        let selfR = NaN;
        enemyQ.length = 0;
        enemyR.length = 0;
        grid.forEachCell((cell, hex) => {
            if (cell.occupantKind !== OccupantKind.Unit) return;
            const unitEid = cell.occupantEid!;
            if (unitEid === selfEid) {
                selfQ = hex.q;
                selfR = hex.r;
            } else if (TeamRef.id[unitEid] !== myTeam) {
                enemyQ.push(hex.q);
                enemyR.push(hex.r);
            }
        });
        if (Number.isNaN(selfQ)) continue; // not on the grid (mid-transition) — keep last snapshot

        UnknownInputBoard.reset(selfEid);

        // Pass 2: fill the egocentric window.
        for (let dr = -VIEW_RADIUS; dr <= VIEW_RADIUS; dr++) {
            for (let dq = -VIEW_RADIUS; dq <= VIEW_RADIUS; dq++) {
                // Enemy heat — pure axial math, so it works on off-map/out-of-view
                // cells too (the gradient toward a hidden enemy is the point).
                let heat = 0;
                for (let e = 0; e < enemyQ.length; e++) {
                    const d = hexDeltaDistance(selfQ + dq - enemyQ[e], selfR + dr - enemyR[e]);
                    heat = Math.max(heat, 1 - d / MAX_MAP_DIST);
                }
                if (heat > 0) UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.EnemyHeat, heat);

                const visible = hexDeltaDistance(dq, dr) <= VIEW_RADIUS;
                const cell = visible ? grid.getCell(selfQ + dq, selfR + dr) : undefined;
                if (!cell) {
                    // Off-map or beyond the view radius — not enterable, not visible.
                    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Obstacle, 1);
                    continue;
                }

                const kind = cell.occupantKind;
                if (kind === null) continue;

                if (kind === OccupantKind.Obstacle) {
                    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Obstacle, 1);
                    continue;
                }

                if (kind === OccupantKind.Reserved) {
                    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Reserved, 1);
                    continue;
                }

                if (kind === OccupantKind.Unit) {
                    const unitEid = cell.occupantEid!;
                    const plane =
                        unitEid === selfEid
                            ? BoardChannel.Self
                            : TeamRef.id[unitEid] === myTeam
                              ? BoardChannel.Ally
                              : BoardChannel.Enemy;

                    UnknownInputBoard.setDelta(selfEid, dq, dr, plane, 1);
                    UnknownInputBoard.setDelta(selfEid, dq, dr, BoardChannel.Hp, getTankHealth(unitEid));
                }
            }
        }

        // Threat from live enemy bullets — straight-line projection of each
        // bullet's remaining (fixed-distance) flight path onto the window.
        markBulletThreat(selfEid, myTeam, selfQ, selfR, grid, world);
    }
}
