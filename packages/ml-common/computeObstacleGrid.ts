import { query } from 'bitecs';
import type { World } from 'bitecs';
import { Obstacle } from '../tanks/src/Game/ECS/Components/Obstacle.ts';
import { Hitable } from '../tanks/src/Game/ECS/Components/Hitable.ts';
import { RigidBodyState } from '../tanks/src/Game/ECS/Components/Physical.ts';
import { Shape } from '../renderer/src/ECS/Components/Shape.ts';
import { GRID_SIZE } from '../ml/src/Models/Create.ts';
import { abs, cos, sin, max, floor, min } from '../../lib/math.ts';

/**
 * Compute a flat binary obstacle grid for the battlefield.
 * Each cell is 1 if an obstacle overlaps it, 0 otherwise.
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

        const minX = px - aabbHW;
        const maxX = px + aabbHW;
        const minY = py - aabbHH;
        const maxY = py + aabbHH;

        // Map AABB to grid cell range
        const colMin = max(0, floor(minX / cellW));
        const colMax = min(GRID_SIZE - 1, floor(maxX / cellW));
        const rowMin = max(0, floor(minY / cellH));
        const rowMax = min(GRID_SIZE - 1, floor(maxY / cellH));

        for (let row = rowMin; row <= rowMax; row++) {
            for (let col = colMin; col <= colMax; col++) {
                grid[row * GRID_SIZE + col] = 1;
            }
        }
    }

    return grid;
}
