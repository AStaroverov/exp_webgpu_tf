/**
 * markBulletThreat — fill one observer's `UnderFire` board plane.
 *
 * A bullet flies a FIXED distance (`DestroyByDistance.maxDistanceSq`) along a
 * straight line from its `origin`, so its entire remaining path is known. For
 * every live ENEMY bullet (threat is enemy fire, POV-relative), we walk the
 * stretch the bullet has NOT yet crossed — from its current position to its
 * endpoint — and mark each hex it will pass through as under fire (0/1). The
 * bullet's current cell is not special-cased as the threat; the cells AHEAD are.
 *
 * The walk stops at the first cell physically blocking the shot (a `Unit` or
 * `Obstacle`) — the bullet hits it — and at the grid edge. The cell the bullet
 * currently sits in is never treated as a blocker (it just left the muzzle).
 *
 * Pure straight-line geometry over `worldToHex`; no honeycomb allocation.
 */

import { query, World } from 'bitecs';
import { GameDI } from '../../../unknown/src/Game/DI/GameDI.ts';
import { getGameComponents } from '../../../unknown/src/Game/ECS/createGameWorld.ts';
import { HexGridConfig } from '../../../unknown/src/Game/Map/HexConfig.ts';
import { OccupantKind, type HexGrid } from '../../../unknown/src/Game/Map/HexGrid.ts';
import { BoardChannel, UnknownInputBoard } from './board.ts';

/** Sample interval along the ray (px). Below a hex inradius (≈0.866·radius) so no cell is skipped. */
const SAMPLE_STEP = HexGridConfig.radius * 0.5;
/** Below this travelled distance the direction comes from velocity, not (pos − origin). */
const ORIGIN_EPS = 1e-3;

export function markBulletThreat(
    selfEid: number,
    myTeam: number,
    grid: HexGrid,
    world: World = GameDI.world,
): void {
    const { Bullet, RigidBodyState, DestroyByDistance, TeamRef } = getGameComponents(world);
    const bullets = query(world, [Bullet, RigidBodyState, DestroyByDistance, TeamRef]);

    for (let i = 0; i < bullets.length; i++) {
        const eid = bullets[i];
        if (TeamRef.id[eid] === myTeam) continue; // only enemy fire is a threat

        const px = RigidBodyState.position.get(eid, 0);
        const py = RigidBodyState.position.get(eid, 1);
        const ox = DestroyByDistance.origin.get(eid, 0);
        const oy = DestroyByDistance.origin.get(eid, 1);

        const dx = px - ox;
        const dy = py - oy;
        const travelled = Math.hypot(dx, dy);

        // Direction: along (pos − origin) once moving, else along velocity at spawn.
        let dirX: number;
        let dirY: number;
        if (travelled > ORIGIN_EPS) {
            dirX = dx / travelled;
            dirY = dy / travelled;
        } else {
            const vx = RigidBodyState.linvel.get(eid, 0);
            const vy = RigidBodyState.linvel.get(eid, 1);
            const speed = Math.hypot(vx, vy);
            if (speed <= ORIGIN_EPS) continue; // unknown heading → skip
            dirX = vx / speed;
            dirY = vy / speed;
        }

        const maxDist = Math.sqrt(DestroyByDistance.maxDistanceSq[eid]);
        const remaining = maxDist - travelled;
        if (remaining <= 0) continue;

        // Walk the remaining path, marking each distinct hex it crosses.
        let lastKey = -1;
        let firstCell = true;
        for (let t = 0; t <= remaining; t += SAMPLE_STEP) {
            const sx = px + dirX * t;
            const sy = py + dirY * t;
            const hex = grid.worldToHex(sx, sy);
            if (!hex) break; // left the grid — a straight ray won't re-enter

            const key = hex.row * HexGridConfig.cols + hex.col;
            if (key === lastKey) continue; // same cell as previous sample
            lastKey = key;

            UnknownInputBoard.set(selfEid, hex.row, hex.col, BoardChannel.UnderFire, 1);

            // The bullet's current cell never blocks (it just left the muzzle);
            // any later Unit/Obstacle cell stops the shot there.
            if (!firstCell) {
                const cell = grid.getCell(hex.q, hex.r);
                const kind = cell?.occupantKind;
                if (kind === OccupantKind.Unit || kind === OccupantKind.Obstacle) break;
            }
            firstCell = false;
        }
    }
}
