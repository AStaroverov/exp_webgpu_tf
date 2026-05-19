import { query } from 'bitecs';
import type { World } from 'bitecs';
import { Obstacle } from '../../../tanks/src/Game/ECS/Components/Obstacle.ts';
import { Hitable } from '../../../tanks/src/Game/ECS/Components/Hitable.ts';
import { RigidBodyState } from '../../../tanks/src/Game/ECS/Components/Physical.ts';
import { Shape } from '../../../renderer/src/ECS/Components/Shape.ts';
import { GRID_SIZE } from '../models/dims.ts';
import { abs, cos, sin, max, floor, min } from '../../../../lib/math.ts';

/**
 * Compute obstacle density grid for the battlefield.
 * Each cell contains a value in [0, 1] representing
 * what fraction of the cell area is covered by obstacles.
 * Computed once per episode (obstacles are static).
 */
export function computeObstacleGrid(
    world: World,
    width: number,
    height: number,
): Float32Array {
    const grid = new Float32Array(GRID_SIZE * GRID_SIZE);
    const cellW = width / GRID_SIZE;
    const cellH = height / GRID_SIZE;
    const cellArea = cellW * cellH;

    const eids = query(world, [Obstacle, Hitable]);

    for (let i = 0; i < eids.length; i++) {
        const eid = eids[i];

        const px = RigidBodyState.position.get(eid, 0);
        const py = RigidBodyState.position.get(eid, 1);
        const shapeVals = Shape.values.getBatch(eid);
        const hw = shapeVals[0] / 2;  // half-width
        const hh = shapeVals[1] / 2;  // half-height
        const rot = RigidBodyState.rotation[eid];

        // Compute AABB of the rotated rectangle
        const cosR = abs(cos(rot));
        const sinR = abs(sin(rot));
        const aabbHW = hw * cosR + hh * sinR;
        const aabbHH = hw * sinR + hh * cosR;

        const obsMinX = px - aabbHW;
        const obsMaxX = px + aabbHW;
        const obsMinY = py - aabbHH;
        const obsMaxY = py + aabbHH;

        // Ratio of actual obstacle area to AABB area for density correction
        const obstacleArea = shapeVals[0] * shapeVals[1]; // actual rectangle area
        const aabbArea = (aabbHW * 2) * (aabbHH * 2);
        const densityScale = aabbArea > 0 ? obstacleArea / aabbArea : 1;

        // Map AABB to grid cell range
        const colMin = max(0, floor(obsMinX / cellW));
        const colMax = min(GRID_SIZE - 1, floor(obsMaxX / cellW));
        const rowMin = max(0, floor(obsMinY / cellH));
        const rowMax = min(GRID_SIZE - 1, floor(obsMaxY / cellH));

        for (let row = rowMin; row <= rowMax; row++) {
            const cellTop = row * cellH;
            const cellBottom = cellTop + cellH;
            const overlapY = max(0, min(obsMaxY, cellBottom) - max(obsMinY, cellTop));

            for (let col = colMin; col <= colMax; col++) {
                const cellLeft = col * cellW;
                const cellRight = cellLeft + cellW;
                const overlapX = max(0, min(obsMaxX, cellRight) - max(obsMinX, cellLeft));

                const overlapArea = overlapX * overlapY * densityScale;
                const idx = row * GRID_SIZE + col;
                grid[idx] = min(1, grid[idx] + overlapArea / cellArea);
            }
        }
    }

    return grid;
}
