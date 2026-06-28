import { NestedArray } from "../../utils.ts";
import { delegate } from "../../../../ECS/src/delegate.ts";
import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../ECS/src/component.ts";

export const ROPE_BUFFER_LENGTH = 100;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

export const createRopeComponent = defineComponent((Rope, { obs }) => {
  const points = NestedArray.f64(ROPE_BUFFER_LENGTH, delegate.defaultSize);
  return {
    points,
    addComponent(world: World, eid: number, pts: ArrayLike<number> = []) {
      addComponent(world, eid, Rope);
      points.getBatch(eid).fill(0);
      points.setBatch(eid, pts);
    },
    set$: obs((eid: number, pts: ArrayLike<number>) => {
      points.setBatch(eid, pts);
    }),
  };
});
