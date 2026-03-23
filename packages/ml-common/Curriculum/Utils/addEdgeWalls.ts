import { createBuilding } from '../../../tanks/src/Game/ECS/Entities/Building/index.ts';

const WALL_THICKNESS = 30;

/**
 * Add solid walls along the 4 edges of the battlefield.
 * Ensures every scenario has obstacles from training start,
 * so the network never learns to ignore the obstacle grid input.
 */
export function addEdgeWalls(width: number, height: number) {
    const cols = Math.ceil(width / WALL_THICKNESS);

    // Top wall
    createBuilding({
        x: width / 2,
        y: 0,
        cols,
        rows: 1,
        cellSize: WALL_THICKNESS,
        wallThickness: WALL_THICKNESS,
        destructionThreshold: 1,
        interiorWallChance: 0,
        noiseScale: 0,
    });

    // Bottom wall
    createBuilding({
        x: width / 2,
        y: height,
        cols,
        rows: 1,
        cellSize: WALL_THICKNESS,
        wallThickness: WALL_THICKNESS,
        destructionThreshold: 1,
        interiorWallChance: 0,
        noiseScale: 0,
    });

    const rows = Math.ceil(height / WALL_THICKNESS);

    // Left wall
    createBuilding({
        x: 0,
        y: height / 2,
        cols: 1,
        rows,
        cellSize: WALL_THICKNESS,
        wallThickness: WALL_THICKNESS,
        destructionThreshold: 1,
        interiorWallChance: 0,
        noiseScale: 0,
    });

    // Right wall
    createBuilding({
        x: width,
        y: height / 2,
        cols: 1,
        rows,
        cellSize: WALL_THICKNESS,
        wallThickness: WALL_THICKNESS,
        destructionThreshold: 1,
        interiorWallChance: 0,
        noiseScale: 0,
    });
}
