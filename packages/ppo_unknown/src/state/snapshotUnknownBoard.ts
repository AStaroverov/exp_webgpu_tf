/**
 * snapshotUnknownBoard — the unknown-game analogue of tanks' `snapshotTankInputTensor`.
 *
 * Fills every learning tank's `UnknownInputBoard` with a chess-like, POV-relative
 * view of the hex grid. Occupancy is read straight off `MapDI.grid` (kept in sync by
 * `createGridOccupancySystem`), so this touches NO physics — exactly the strategic
 * principle: the board is the position, the planes are the pieces.
 *
 * Per observer:
 *   - Obstacle cells           → `Obstacle` plane.
 *   - The observer's own cell  → `Self` plane (+ its hp).
 *   - Same-team unit cells     → `Ally`  plane (+ hp).
 *   - Other-team unit cells    → `Enemy` plane (+ hp).
 *   - `Reserved` cells         → ignored (transient movement reservation, dropped).
 *
 * Prereq: each observing tank must have `UnknownInputBoard` added (agent setup calls
 * `UnknownInputBoard.addComponent(world, tankEid)`).
 */

import { query } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { MapDI } from '../../../unknown/src/Game/DI/MapDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { OccupantKind } from '../../../unknown/src/Game/Map/HexGrid.ts';
import { getTankHealth } from '../../../unknown/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { BoardChannel, UnknownInputBoard } from './board.ts';

export function snapshotUnknownBoard({ world } = GameDI) {
    const grid = MapDI.grid;
    if (!grid) return;

    const { Tank, Vehicle, TeamRef } = getGameComponents(world);
    const observers = query(world, [Vehicle, Tank, UnknownInputBoard]);

    for (let i = 0; i < observers.length; i++) {
        const selfEid = observers[i];
        const myTeam = TeamRef.id[selfEid];

        UnknownInputBoard.reset(selfEid);

        grid.forEachCell((cell, hex) => {
            const kind = cell.occupantKind;
            if (kind === null) return;

            if (kind === OccupantKind.Obstacle) {
                UnknownInputBoard.set(selfEid, hex.row, hex.col, BoardChannel.Obstacle, 1);
                return;
            }

            if (kind === OccupantKind.Unit) {
                const unitEid = cell.occupantEid!;
                const plane =
                    unitEid === selfEid
                        ? BoardChannel.Self
                        : TeamRef.id[unitEid] === myTeam
                          ? BoardChannel.Ally
                          : BoardChannel.Enemy;

                UnknownInputBoard.set(selfEid, hex.row, hex.col, plane, 1);
                UnknownInputBoard.set(selfEid, hex.row, hex.col, BoardChannel.Hp, getTankHealth(unitEid));
            }
        });
    }
}
