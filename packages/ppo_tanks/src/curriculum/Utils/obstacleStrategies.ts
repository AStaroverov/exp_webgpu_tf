import { PI } from '../../../../../lib/math.ts';
import { randomRangeInt } from '../../../../../lib/random.ts';
import {
    CellContent,
    getCellWorldPosition,
    getSpawnGrid,
    isCellEmpty,
    setCellContent,
} from '../../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createBuilding } from '../../../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { addFauna } from './addFauna.ts';
import { getLastDiagonalGeometry } from './spawnStrategies.ts';

export type ObstacleStrategy =
    | { kind: 'none' }
    | { kind: 'fauna' }
    | { kind: 'single-building' }
    | { kind: 'diagonal-wall' };

export function applyObstacles(strategy: ObstacleStrategy): void {
    switch (strategy.kind) {
        case 'none':
            return;
        case 'fauna':
            addFauna();
            return;
        case 'single-building':
            placeSingleBuilding();
            return;
        case 'diagonal-wall':
            placeDiagonalWall();
            return;
    }
}

function placeSingleBuilding(): void {
    // Obstacles run after spawn in createScenario, so filter out cells already
    // occupied by tanks. Legacy createRandomNvsMScenario placed the building first
    // to reserve the cell; this restores the same no-overlap invariant in reverse.
    const grid = getSpawnGrid();
    const freeInterior: { col: number; row: number }[] = [];
    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            if (isCellEmpty(col, row)) freeInterior.push({ col, row });
        }
    }
    if (freeInterior.length === 0) return;
    const cell = freeInterior[randomRangeInt(0, freeInterior.length - 1)];
    const pos = getCellWorldPosition(cell.col, cell.row);
    createBuilding({ x: pos.x, y: pos.y });
    setCellContent(cell.col, cell.row, CellContent.Obstacle);
}

function placeDiagonalWall(): void {
    // Requires diagonal spawn to have run first so the wall aligns with the spawn diagonal.
    const geometry = getLastDiagonalGeometry();
    if (geometry == null) return;

    const buildingSpacing = 150;
    const perpAngle = geometry.baseDiagonalAngle + PI / 2;
    const perpX = Math.cos(perpAngle);
    const perpY = Math.sin(perpAngle);
    const random = Math.random();

    if (random < 0.33) {
        for (const offset of [0, 1]) {
            createBuilding({
                x: geometry.centerX + perpX * offset * buildingSpacing,
                y: geometry.centerY + perpY * offset * buildingSpacing,
            });
        }
    } else {
        const direction = Math.random() < 0.5 ? -1 : 1;
        for (const offset of [0, 1, 2, 3, 4]) {
            createBuilding({
                x: geometry.centerX + perpX * offset * direction * buildingSpacing,
                y: geometry.centerY + perpY * offset * direction * buildingSpacing,
            });
        }
    }
}
