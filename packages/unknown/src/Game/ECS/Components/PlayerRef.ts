import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createPlayerRefComponent = defineComponent((PlayerRef) => {
  const id = TypedArray.u32(delegate.defaultSize);
  return {
    id,
    addComponent(world: World, eid: EntityId, playerId: number) {
      addComponent(world, eid, PlayerRef);
      id[eid] = playerId;
    },
  };
});
