import { addEntity } from 'bitecs';
import { addTransformComponents, applyMatrixTranslate, LocalTransform, setMatrixScale } from 'renderer/src/ECS/Components/Transform.ts';
import { Color, Roundness } from 'renderer/src/ECS/Components/Common.ts';
import { Shape, ShapeKind } from 'renderer/src/ECS/Components/Shape.ts';

import { Cell, CellState, GRID_SIZE } from '../Components/Cell.js';
import { GameDI } from '../../DI/GameDI.js';

// Cell visual size in world units (affects how big cells appear)
export const CELL_SIZE = 20;
export const CELL_PADDING = 2;

// Colors for different cell states
const COLOR_SPAWNING: [number, number, number, number] = [0.2, 0.7, 0.3, 0.5];
const COLOR_ALIVE: [number, number, number, number] = [0.1, 0.8, 0.2, 1.0];
const COLOR_DYING: [number, number, number, number] = [0.8, 0.2, 0.1, 0.7];

export function gridToWorld(gridX: number, gridY: number): [number, number] {
    return [gridX * CELL_SIZE, gridY * CELL_SIZE];
}

export function worldToGrid(worldX: number, worldY: number): [number, number] {
    return [Math.round(worldX / CELL_SIZE), Math.round(worldY / CELL_SIZE)];
}

export function createLifeCell(gridX: number, gridY: number, state: CellState = CellState.SPAWNING, { world } = GameDI): number {
    // Don't create if position is already occupied
    if (Cell.getEntityAt(gridX, gridY) !== 0) {
        return 0;
    }

    // Don't create outside grid bounds
    if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
        return 0;
    }

    const eid = addEntity(world);

    // Add cell component
    Cell.addComponent(world, eid, gridX, gridY, state);

    // Add transform for rendering
    addTransformComponents(world, eid);
    const [worldX, worldY] = gridToWorld(gridX, gridY);
    applyMatrixTranslate(LocalTransform.matrix.getBatch(eid), worldX, worldY, 0.5);

    // Add shape (rectangle for cell)
    const size = CELL_SIZE - CELL_PADDING;
    Shape.addComponent(world, eid, ShapeKind.Rectangle, size, size);
    
    // Add color based on state
    const color = state === CellState.SPAWNING ? COLOR_SPAWNING 
        : state === CellState.ALIVE ? COLOR_ALIVE 
        : COLOR_DYING;
    Color.addComponent(world, eid, color[0], color[1], color[2], color[3]);
    
    // Add roundness for nicer look
    Roundness.addComponent(world, eid, 2);

    return eid;
}

export function updateCellVisuals(eid: number) {
    const state = Cell.state[eid];
    const progress = Cell.getSpawnProgress(eid);

    if (state === CellState.SPAWNING) {
        // Fade in and grow while spawning
        const alpha = 0.3 + progress * 0.7;
        const scale = 0.3 + progress * 0.7;
        Color.set$(eid, COLOR_SPAWNING[0], COLOR_SPAWNING[1], COLOR_SPAWNING[2], alpha);
        setMatrixScale(LocalTransform.matrix.getBatch(eid), scale, scale);
    } else if (state === CellState.ALIVE) {
        Color.set$(eid, COLOR_ALIVE[0], COLOR_ALIVE[1], COLOR_ALIVE[2], COLOR_ALIVE[3]);
        setMatrixScale(LocalTransform.matrix.getBatch(eid), 1, 1);
    } else if (state === CellState.DYING) {
        Color.set$(eid, COLOR_DYING[0], COLOR_DYING[1], COLOR_DYING[2], COLOR_DYING[3]);
    }
}

