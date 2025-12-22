import { GameDI } from '../../../Game/DI/GameDI.ts';

export const GRID_CELL_SIZE = 200;

export enum CellContent {
    Empty = 0,
    Vehicle = 1,
    Structure = 2,
    Obstacle = 3,
}

export type GridCell = {
    content: CellContent;
    entityId?: number;
};

export type SpawnGrid = {
    cells: GridCell[][];
    cols: number;
    rows: number;
    cellSize: number;
};

let grid: SpawnGrid | null = null;

export function createSpawnGrid(): SpawnGrid {
    const cols = Math.floor(GameDI.width / GRID_CELL_SIZE);
    const rows = Math.floor(GameDI.height / GRID_CELL_SIZE);

    const cells: GridCell[][] = [];
    for (let row = 0; row < rows; row++) {
        cells[row] = [];
        for (let col = 0; col < cols; col++) {
            cells[row][col] = { content: CellContent.Empty };
        }
    }

    grid = { cells, cols, rows, cellSize: GRID_CELL_SIZE };
    return grid;
}

export function getSpawnGrid(): SpawnGrid {
    if (!grid) {
        return createSpawnGrid();
    }
    return grid;
}

export function resetSpawnGrid(): void {
    grid = null;
}

export function getCellWorldPosition(col: number, row: number): { x: number; y: number } {
    const cellSize = GRID_CELL_SIZE;
    return {
        x: col * cellSize + cellSize / 2,
        y: row * cellSize + cellSize / 2,
    };
}

export function getGridCellFromWorld(x: number, y: number): { col: number; row: number } {
    return {
        col: Math.floor(x / GRID_CELL_SIZE),
        row: Math.floor(y / GRID_CELL_SIZE),
    };
}

export function setCellContent(col: number, row: number, content: CellContent, entityId?: number): void {
    const g = getSpawnGrid();
    if (col >= 0 && col < g.cols && row >= 0 && row < g.rows) {
        g.cells[row][col] = { content, entityId };
    }
}

export function getCellContent(col: number, row: number): GridCell | null {
    const g = getSpawnGrid();
    if (col >= 0 && col < g.cols && row >= 0 && row < g.rows) {
        return g.cells[row][col];
    }
    return null;
}

export function isCellEmpty(col: number, row: number): boolean {
    const cell = getCellContent(col, row);
    return cell !== null && cell.content === CellContent.Empty;
}

/**
 * Get spawn column for a team.
 * Team 0 spawns on the left edge (col 0).
 * Team 1 spawns on the right edge (last col).
 */
export function getTeamSpawnColumn(teamId: number): number {
    const g = getSpawnGrid();
    return teamId === 0 ? 0 : g.cols - 1;
}

/**
 * Get the center row of the grid (for player spawn).
 */
export function getCenterRow(): number {
    const g = getSpawnGrid();
    return Math.floor(g.rows / 2);
}

/**
 * Generate spawn positions for a team.
 * First slot (index 0) is for the player - spawns at the center.
 * Subsequent slots alternate above and below the player.
 * 
 * @param teamId - 0 for left team, 1 for right team
 * @param slotIndex - 0 for player, 1+ for other units
 * @returns Grid position { col, row } and world position { x, y }
 */
export function getTeamSpawnPosition(
    teamId: number,
    slotIndex: number
): { col: number; row: number; x: number; y: number } {
    const col = getTeamSpawnColumn(teamId);
    const centerRow = getCenterRow();

    let row: number;
    if (slotIndex === 0) {
        // Player slot - center
        row = centerRow;
    } else {
        // Alternate above/below center
        // slotIndex 1 -> +1 (below), slotIndex 2 -> -1 (above)
        // slotIndex 3 -> +2 (below), slotIndex 4 -> -2 (above)
        const offset = Math.ceil(slotIndex / 2);
        const direction = slotIndex % 2 === 1 ? 1 : -1;
        row = centerRow + offset * direction;
    }

    // Clamp row to valid range
    const g = getSpawnGrid();
    row = Math.max(0, Math.min(g.rows - 1, row));

    const { x, y } = getCellWorldPosition(col, row);
    return { col, row, x, y };
}

/**
 * Allocate a spawn cell for an entity.
 * Marks the cell as occupied and returns the position.
 */
export function allocateSpawnCell(
    teamId: number,
    slotIndex: number,
    content: CellContent,
    entityId?: number
): { col: number; row: number; x: number; y: number } {
    const pos = getTeamSpawnPosition(teamId, slotIndex);
    setCellContent(pos.col, pos.row, content, entityId);
    return pos;
}

/**
 * Find the next available spawn slot for a team.
 * Returns the slot index or -1 if no slots available.
 */
export function findNextAvailableSlot(teamId: number, maxSlots: number = 10): number {
    for (let slotIndex = 0; slotIndex < maxSlots; slotIndex++) {
        const pos = getTeamSpawnPosition(teamId, slotIndex);
        if (isCellEmpty(pos.col, pos.row)) {
            return slotIndex;
        }
    }
    return -1;
}

