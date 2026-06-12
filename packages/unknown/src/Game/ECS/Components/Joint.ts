import { addComponent, hasComponent, removeComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createJointComponent = defineComponent((Joint, ctx) => {
  const pid = ctx.table.flat(Float64Array);
  return {
    pid,
    addComponent(world: World, eid: EntityId, p: number) {
      addComponent(world, eid, Joint);
      pid.set(eid, p);
    },
    removeComponent(world: World, eid: EntityId) {
      removeComponent(world, eid, Joint);
    },
    setPid(eid: EntityId, p: number) {
      pid.set(eid, p);
    },
    resetComponent(eid: EntityId) {
      if (hasComponent(ctx.world, eid, Joint)) pid.set(eid, 0);
    },
  };
});
