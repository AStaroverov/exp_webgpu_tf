import { TypedArray } from "renderer/src/utils.ts";
import { delegate } from "renderer/src/delegate.ts";
import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createDestroyComponent = defineComponent((Destroy) => {
  const recursive = TypedArray.i8(delegate.defaultSize);
  return {
    recursive,
    addComponent(world: World, eid: number, rec: boolean = true) {
      addComponent(world, eid, Destroy);
      recursive[eid] = rec ? 1 : 0;
    },
  };
});

export const createDestroyByTimeoutComponent = defineComponent((DestroyByTimeout, ctx) => {
  const timeout = ctx.table.flat(Float64Array);
  return {
    timeout,
    addComponent(world: World, eid: number, t: number) {
      addComponent(world, eid, DestroyByTimeout);
      timeout.set(eid, t);
    },
    updateTimeout(eid: number, delta: number) {
      timeout.set(eid, timeout.get(eid) - delta);
    },
    resetTimeout(eid: number, t: number) {
      timeout.set(eid, t);
    },
  };
});

export const createDestroyByDistanceComponent = defineComponent((DestroyByDistance, ctx) => {
  // Origin point the distance is measured from.
  const origin = ctx.table.nested(Float64Array, 2);
  // Squared max distance — avoids a sqrt per entity per tick.
  const maxDistanceSq = ctx.table.flat(Float64Array);
  return {
    origin,
    maxDistanceSq,
    addComponent(world: World, eid: number, x: number, y: number, maxDistance: number) {
      addComponent(world, eid, DestroyByDistance);
      origin.set(eid, 0, x);
      origin.set(eid, 1, y);
      maxDistanceSq.set(eid, maxDistance * maxDistance);
    },
  };
});
