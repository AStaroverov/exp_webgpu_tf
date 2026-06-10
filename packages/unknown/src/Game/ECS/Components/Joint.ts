import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { addComponent, EntityId, removeComponent, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createJointComponent = defineComponent((Joint) => {
  const pid = TypedArray.f64(delegate.defaultSize);
  return {
    pid,
    addComponent(world: World, eid: EntityId, p: number) {
      addComponent(world, eid, Joint);
      pid[eid] = p;
    },
    removeComponent(world: World, eid: EntityId) {
      removeComponent(world, eid, Joint);
      pid[eid] = 0;
    },
    setPid(eid: EntityId, p: number) {
      pid[eid] = p;
    },
    resetComponent(eid: EntityId) {
      pid[eid] = 0;
    },
  };
});
