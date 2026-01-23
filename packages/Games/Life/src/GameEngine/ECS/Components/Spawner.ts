import { addComponent, World } from 'bitecs';
import { TypedArray } from 'renderer/src/utils.ts';
import { delegate } from 'renderer/src/delegate.ts';
import { component } from 'renderer/src/ECS/utils.ts';

export const Spawner = component({
    // Grid position
    gridX: TypedArray.i32(delegate.defaultSize),
    gridY: TypedArray.i32(delegate.defaultSize),
    
    // Spawn interval in ms
    interval: TypedArray.f32(delegate.defaultSize),
    // Time until next spawn
    timer: TypedArray.f32(delegate.defaultSize),
    // Number of cells to spawn (0 = infinite)
    count: TypedArray.i32(delegate.defaultSize),

    addComponent(world: World, eid: number, gridX: number, gridY: number, interval: number = 500, count: number = 0) {
        addComponent(world, eid, Spawner);
        Spawner.gridX[eid] = gridX;
        Spawner.gridY[eid] = gridY;
        Spawner.interval[eid] = interval;
        Spawner.timer[eid] = 0; // Spawn immediately
        Spawner.count[eid] = count;
    },

    updateTimer(eid: number, delta: number) {
        Spawner.timer[eid] -= delta;
    },

    resetTimer(eid: number) {
        Spawner.timer[eid] = Spawner.interval[eid];
    },

    shouldSpawn(eid: number): boolean {
        return Spawner.timer[eid] <= 0;
    },

    decrementCount(eid: number) {
        if (Spawner.count[eid] > 0) {
            Spawner.count[eid]--;
        }
    },

    isDepleted(eid: number): boolean {
        return Spawner.count[eid] === 0;
    }
});

