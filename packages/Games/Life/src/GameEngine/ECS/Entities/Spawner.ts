import { addEntity, World } from 'bitecs';
import { Spawner } from '../Components/Spawner.js';
import { addTransformComponents, applyMatrixTranslate, LocalTransform } from 'renderer/src/ECS/Components/Transform.ts';
import { Color, Roundness } from 'renderer/src/ECS/Components/Common.ts';
import { Shape, ShapeKind } from 'renderer/src/ECS/Components/Shape.ts';
import { CELL_SIZE, gridToWorld } from './Cell.js';
import { GRID_SIZE } from '../Components/Cell.js';

// Spawner visual appearance
const SPAWNER_COLOR: [number, number, number, number] = [0.9, 0.6, 0.1, 0.9];
const SPAWNER_SIZE = CELL_SIZE * 0.7;

export function createSpawner(world: World, gridX: number, gridY: number, interval: number = 500): number {
    // Don't create outside grid bounds
    if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
        return 0;
    }

    const eid = addEntity(world);

    // Add spawner component
    Spawner.addComponent(world, eid, gridX, gridY, interval);

    // Add transform for rendering
    addTransformComponents(world, eid);
    const [worldX, worldY] = gridToWorld(gridX, gridY);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), worldX, worldY, 1); // Slightly above cells

    // Add shape (diamond/rhombus for spawner)
    Shape.addComponent(world, eid, ShapeKind.Circle, SPAWNER_SIZE);
    
    // Add color
    Color.addComponent(world, eid, SPAWNER_COLOR[0], SPAWNER_COLOR[1], SPAWNER_COLOR[2], SPAWNER_COLOR[3]);
    
    // Add roundness
    Roundness.addComponent(world, eid, 1);

    return eid;
}

