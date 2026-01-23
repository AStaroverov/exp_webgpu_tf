import { World, EntityId, addComponent, removeComponent } from "bitecs";
import { delegate } from "renderer/src/delegate";
import { component } from "renderer/src/ECS/utils";
import { NestedArray, TypedArray } from "renderer/src/utils";

export const GRID_SIZE = 300;

export enum CellState {
    SPAWNING = 0,  // Just created, transitioning to life
    ALIVE = 1,     // Living cell
    DYING = 2,     // Marked for death
}

export const Cell = component({
    // Grid position
    gridX: TypedArray.i32(delegate.defaultSize),
    gridY: TypedArray.i32(delegate.defaultSize),

    // Cell state
    state: TypedArray.u8(delegate.defaultSize),
    
    // Age in ms (for visual effects)
    age: TypedArray.f32(delegate.defaultSize),
    
    // Time to transition from SPAWNING to ALIVE
    spawnTime: TypedArray.f32(delegate.defaultSize),

    // Global grid to track which cells are alive at which position
    // Stores entity ID at each position (0 = empty)
    grid: NestedArray.f64(GRID_SIZE, GRID_SIZE),

    addComponent(world: World, eid: EntityId, gridX: number, gridY: number, state: CellState = CellState.SPAWNING, spawnTime: number = 300) {
        addComponent(world, eid, Cell);
        Cell.gridX[eid] = gridX;
        Cell.gridY[eid] = gridY;
        Cell.state[eid] = state;
        Cell.age[eid] = 0;
        Cell.spawnTime[eid] = spawnTime;
        Cell.grid.set(gridX, gridY, eid);
    },

    removeComponent(world: World, eid: EntityId) {
        const x = Cell.gridX[eid];
        const y = Cell.gridY[eid];
        Cell.grid.set(x, y, 0);
        Cell.gridX[eid] = 0;
        Cell.gridY[eid] = 0;
        Cell.state[eid] = 0;
        Cell.age[eid] = 0;
        removeComponent(world, eid, Cell);
    },

    updateAge(eid: EntityId, delta: number) {
        Cell.age[eid] += delta;
    },

    isAlive(eid: EntityId): boolean {
        return Cell.state[eid] === CellState.ALIVE;
    },

    isSpawning(eid: EntityId): boolean {
        return Cell.state[eid] === CellState.SPAWNING;
    },

    isDying(eid: EntityId): boolean {
        return Cell.state[eid] === CellState.DYING;
    },

    getSpawnProgress(eid: EntityId): number {
        return Math.min(1, Cell.age[eid] / Cell.spawnTime[eid]);
    },

    setAlive(eid: EntityId) {
        Cell.state[eid] = CellState.ALIVE;
    },

    setDying(eid: EntityId) {
        Cell.state[eid] = CellState.DYING;
    },

    // Check if a grid position is occupied by an ALIVE cell
    isGridPositionAlive(gridX: number, gridY: number): boolean {
        if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
            return false;
        }
        const eid = Cell.grid.get(gridX, gridY);
        if (eid === 0) return false;
        return Cell.state[eid] === CellState.ALIVE;
    },

    // Get entity ID at grid position
    getEntityAt(gridX: number, gridY: number): EntityId {
        if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) {
            return 0 as EntityId;
        }
        return Cell.grid.get(gridX, gridY) as EntityId;
    },

    // Count alive neighbors
    countAliveNeighbors(gridX: number, gridY: number): number {
        let count = 0;
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                if (dx === 0 && dy === 0) continue;
                if (Cell.isGridPositionAlive(gridX + dx, gridY + dy)) {
                    count++;
                }
            }
        }
        return count;
    },
});
