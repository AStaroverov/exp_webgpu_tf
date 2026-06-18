import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createPlayerRefComponent = defineComponent((PlayerRef, ctx) => {
  const id = ctx.table.flat(Uint32Array);
  return {
    id,
    addComponent(world: World, eid: EntityId, playerId: number) {
      addComponent(world, eid, PlayerRef);
      id.set(eid, playerId);
    },
  };
});
