import {
    getSpawnGrid,
    isCellEmpty,
    getCellWorldPosition,
    setCellContent,
    CellContent,
} from '../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createRock } from '../../../tanks/src/Game/ECS/Entities/Rock/Rock.ts';

export type AddFaunaOptions = {
    rockProbability?: number;
};

export function addFauna(options?: AddFaunaOptions) {
    const grid = getSpawnGrid();
    const rockProbability = options?.rockProbability ?? 0.3;

    // Skip edge rows/columns to keep spawn areas clear
    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            if (!isCellEmpty(col, row)) continue;
            
            if (Math.random() < rockProbability) {
                const { x, y } = getCellWorldPosition(col, row);
                createRock({ x, y });
                setCellContent(col, row, CellContent.Obstacle);
            }
        }
    }
}

