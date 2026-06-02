/**
 * spawnObstacles — scatter rocks on the hex grid in three phases:
 *
 *   PREBUILD  build a list of ObstaclePlan claims on a *virtual* occupancy set
 *             (never touches ECS or the real grid), so attempts are cheap.
 *   VALIDATE  check the free cells stay one connected region (no walled-off
 *             pockets that would trap units); re-roll the whole layout if not.
 *   COMMIT    create the entities and occupy the real grid for the accepted plans.
 *
 * Planning depends only on the HexGrid (pure data), so it works for any world;
 * commit takes `world` and writes occupancy into it.
 */

import { GameDI } from '../../../DI/GameDI.ts';
import { MapDI } from '../../../DI/MapDI.ts';
import { HexGrid } from '../../../Map/HexGrid.ts';
import { ObstacleConfig } from '../../../Config/index.ts';
import { createRock } from './Rocks/Rock.ts';
import { ObstaclePlan } from './types.ts';

const cellKey = (q: number, r: number): string => `${q},${r}`;

export function spawnObstacles({ grid } = MapDI, { world } = GameDI): void {
    let plans: ObstaclePlan[] = [];

    for (let attempt = 0; attempt < ObstacleConfig.maxLayoutAttempts; attempt++) {
        plans = prebuild(grid);
        if (validate(grid, plans)) break;
        // Last attempt falls through and commits anyway (prototype fallback).
        if (attempt === ObstacleConfig.maxLayoutAttempts - 1) {
            console.warn('[spawnObstacles] layout never validated; committing last attempt');
        }
    }

    for (const plan of plans) {
        createRock(plan, { world });
    }
}

/** Build rock claims over a virtual occupancy set seeded with blocked cells. */
function prebuild(grid: HexGrid): ObstaclePlan[] {
    const reserved = new Set<string>();
    grid.forEachCell((c) => {
        if (!grid.isPassable(c.q, c.r)) reserved.add(cellKey(c.q, c.r));
    });

    const plans: ObstaclePlan[] = [];
    grid.forEachCell((cell) => {
        const key = cellKey(cell.q, cell.r);
        if (reserved.has(key)) return;
        if (Math.random() >= ObstacleConfig.spawnChance) return;

        const anchor = { q: cell.q, r: cell.r };
        reserved.add(key);
        plans.push({ anchor, cells: [anchor] });
    });

    return plans;
}

/** The free cells (those not reserved by any plan) must stay one connected region. */
function validate(grid: HexGrid, plans: ObstaclePlan[]): boolean {
    const reserved = new Set<string>();
    grid.forEachCell((c) => {
        if (!grid.isPassable(c.q, c.r)) reserved.add(cellKey(c.q, c.r));
    });
    for (const plan of plans) {
        for (const c of plan.cells) reserved.add(cellKey(c.q, c.r));
    }

    const free: Array<{ q: number; r: number }> = [];
    grid.forEachCell((c) => {
        if (!reserved.has(cellKey(c.q, c.r))) free.push({ q: c.q, r: c.r });
    });
    if (free.length === 0) return true;

    // Flood-fill from the first free cell across free neighbors.
    const seen = new Set<string>([cellKey(free[0].q, free[0].r)]);
    const queue = [free[0]];
    while (queue.length > 0) {
        const cur = queue.pop()!;
        for (const n of grid.neighbors(cur)) {
            const nk = cellKey(n.q, n.r);
            if (reserved.has(nk) || seen.has(nk)) continue;
            seen.add(nk);
            queue.push({ q: n.q, r: n.r });
        }
    }

    return seen.size === free.length;
}
