import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.js';
import { Spawner } from '../Components/Spawner.js';
import { createLifeCell } from '../Entities/Cell.js';
import { CellState } from '../Components/Cell.js';

// Global spawn interval in ms
const GLOBAL_SPAWN_INTERVAL = 2000;

export function createSpawnerSystem({ world } = GameDI) {
    // Global timer shared by all spawners
    let globalTimer = 0;

    return (delta: number) => {
        const spawners = query(world, [Spawner]);
        if (spawners.length === 0) return;

        // Update global timer
        globalTimer += delta;

        // Check if it's time to spawn
        if (globalTimer >= GLOBAL_SPAWN_INTERVAL) {
            globalTimer = 0;

            // Spawn for all spawners at once
            for (let i = 0; i < spawners.length; i++) {
                const eid = spawners[i];
                const gridX = Spawner.gridX[eid];
                const gridY = Spawner.gridY[eid];

                // Create a new cell at the spawner position
                createLifeCell(gridX, gridY, CellState.SPAWNING);
            }
        }
    };
}

