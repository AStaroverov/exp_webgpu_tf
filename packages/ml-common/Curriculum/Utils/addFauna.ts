import {
    getSpawnGrid,
    isCellEmpty,
    getCellWorldPosition,
    setCellContent,
    CellContent,
} from '../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createRock } from '../../../tanks/src/Game/ECS/Entities/Rock/Rock.ts';
import { createBuilding } from '../../../tanks/src/Game/ECS/Entities/Building/index.ts';

export type AddFaunaOptions = {
    rockProbability?: number;
    buildingProbability?: number;
};

export function addFauna(options?: AddFaunaOptions) {
    const grid = getSpawnGrid();
    const buildingProbability = options?.buildingProbability ?? 0.05;
    const rockProbability = options?.rockProbability ?? 0.3;

    // Skip edge rows/columns to keep spawn areas clear
    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            if (!isCellEmpty(col, row)) continue;
            
            const rand = Math.random();
            if (rand < buildingProbability) {
                const { x, y } = getCellWorldPosition(col, row);
                createBuilding({ x, y });
                setCellContent(col, row, CellContent.Obstacle);
            } else if (rand < buildingProbability + rockProbability) {
                const { x, y } = getCellWorldPosition(col, row);
                createRock({ x, y });
                setCellContent(col, row, CellContent.Obstacle);
            }
        }
    }
}

