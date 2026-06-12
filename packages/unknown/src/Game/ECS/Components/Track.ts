import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export enum TrackSide {
  Left = 0,
  Right = 1,
}

export const createTrackComponent = defineComponent((Track, ctx) => {
  const side = ctx.table.flat(Int8Array);
  const length = ctx.table.flat(Float64Array);
  return {
    side,
    length,
    addComponent(world: World, eid: EntityId, s: TrackSide, len: number) {
      addComponent(world, eid, Track);
      length.set(eid, len);
      side.set(eid, s);
    },
  };
});
