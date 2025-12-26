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

    // Collect available cells (skip edge rows/columns to keep spawn areas clear)
    const availableCells: { col: number; row: number }[] = [];
    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            if (isCellEmpty(col, row)) {
                availableCells.push({ col, row });
            }
        }
    }

    // Shuffle available cells
    for (let i = availableCells.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [availableCells[i], availableCells[j]] = [availableCells[j], availableCells[i]];
    }

    // Guarantee at least one building and one rock
    const guaranteedCells = new Set<number>();
    if (availableCells.length >= 1) {
        const { col, row } = availableCells[0];
        const { x, y } = getCellWorldPosition(col, row);
        createBuilding({ x, y });
        setCellContent(col, row, CellContent.Obstacle);
        guaranteedCells.add(0);
    }
    if (availableCells.length >= 2) {
        const { col, row } = availableCells[1];
        const { x, y } = getCellWorldPosition(col, row);
        createRock({ x, y });
        setCellContent(col, row, CellContent.Obstacle);
        guaranteedCells.add(1);
    }

    // Place remaining fauna based on probability
    for (let i = 0; i < availableCells.length; i++) {
        if (guaranteedCells.has(i)) continue;
        
        const { col, row } = availableCells[i];
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

