import { World, EntityId, addComponent, removeComponent } from "bitecs";
import { delegate } from "renderer/src/delegate";
import { component } from "renderer/src/ECS/utils";
import { NestedArray, TypedArray } from "renderer/src/utils";

const CELLS_COUNT = 300;

export const Cell = component({
    x: TypedArray.f64(delegate.defaultSize),
    y: TypedArray.f64(delegate.defaultSize),

    grid: NestedArray.f64(CELLS_COUNT, CELLS_COUNT),

    addComponent(world: World, eid: EntityId, x: number, y: number) {
        addComponent(world, eid, Cell);
        Cell.x[eid] = x;
        Cell.y[eid] = y;
        Cell.grid.set(x, y, eid);
    },

    removeComponent(world: World, eid: EntityId) {
        removeComponent(world, eid, Cell);
        Cell.grid.set(Cell.x[eid], Cell.y[eid], 0);
        Cell.x[eid] = 0;
        Cell.y[eid] = 0;
    },
});