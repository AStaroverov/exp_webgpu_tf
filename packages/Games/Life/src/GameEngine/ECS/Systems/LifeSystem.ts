import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.js';
import { Cell, CellState, GRID_SIZE } from '../Components/Cell.js';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.js';
import { createLifeCell, updateCellVisuals } from '../Entities/Cell.js';

// Game of Life simulation interval (ms)
const LIFE_TICK_INTERVAL = 100;

export function createLifeSystem({ world } = GameDI) {
    let timeSinceLastTick = 0;
    
    // Temporary grid to store next generation state
    // -1 = no change, 0 = should die, 1 = should be born, 2 = survives
    const nextState = new Int8Array(GRID_SIZE * GRID_SIZE);

    return (delta: number) => {
        const cells = query(world, [Cell]);

        // Update cell ages and handle spawning state transitions
        for (let i = 0; i < cells.length; i++) {
            const eid = cells[i];
            Cell.updateAge(eid, delta);

            // Transition from SPAWNING to ALIVE when spawn time is complete
            if (Cell.isSpawning(eid) && Cell.getSpawnProgress(eid) >= 1) {
                Cell.setAlive(eid);
            }

            // Update visuals based on state
            updateCellVisuals(eid);
        }

        // Accumulate time for life tick
        timeSinceLastTick += delta;
        if (timeSinceLastTick < LIFE_TICK_INTERVAL) {
            return;
        }
        timeSinceLastTick = 0;

        // Reset next state array
        nextState.fill(-1);

        // Calculate next generation based on Game of Life rules
        // First pass: determine what should happen to each position
        
        // Check all cells for survival
        for (let i = 0; i < cells.length; i++) {
            const eid = cells[i];
            
            // Only apply rules to ALIVE cells
            if (!Cell.isAlive(eid)) continue;

            const gridX = Cell.gridX[eid];
            const gridY = Cell.gridY[eid];
            const neighbors = Cell.countAliveNeighbors(gridX, gridY);

            const idx = gridY * GRID_SIZE + gridX;

            // Rule 1: Any live cell with 2 or 3 neighbors survives
            // Rule 2: Any live cell with fewer than 2 neighbors dies (underpopulation)
            // Rule 3: Any live cell with more than 3 neighbors dies (overpopulation)
            if (neighbors === 2 || neighbors === 3) {
                nextState[idx] = 2; // survives
            } else {
                nextState[idx] = 0; // dies
            }

            // Check neighbors for potential births
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    
                    const nx = gridX + dx;
                    const ny = gridY + dy;
                    
                    if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
                    
                    const nIdx = ny * GRID_SIZE + nx;
                    
                    // Only check empty positions we haven't checked yet
                    if (nextState[nIdx] !== -1) continue;
                    if (Cell.getEntityAt(nx, ny) !== 0) continue;
                    
                    const nNeighbors = Cell.countAliveNeighbors(nx, ny);
                    
                    // Rule 4: Any dead cell with exactly 3 neighbors becomes alive
                    if (nNeighbors === 3) {
                        nextState[nIdx] = 1; // birth
                    }
                }
            }
        }

        // Second pass: apply changes
        // Kill cells
        for (let i = 0; i < cells.length; i++) {
            const eid = cells[i];
            if (!Cell.isAlive(eid)) continue;

            const gridX = Cell.gridX[eid];
            const gridY = Cell.gridY[eid];
            const idx = gridY * GRID_SIZE + gridX;

            if (nextState[idx] === 0) {
                Cell.setDying(eid);
                scheduleRemoveEntity(eid);
            }
        }

        // Birth new cells
        for (let y = 0; y < GRID_SIZE; y++) {
            for (let x = 0; x < GRID_SIZE; x++) {
                const idx = y * GRID_SIZE + x;
                if (nextState[idx] === 1) {
                    // Create new cell directly as ALIVE (skip spawning animation for births)
                    createLifeCell(x, y, CellState.ALIVE);
                }
            }
        }
    };
}
