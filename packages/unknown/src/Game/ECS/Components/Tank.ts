import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { addComponent, EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createTankComponent = defineComponent((Tank) => {
  const turretEId = TypedArray.f64(delegate.defaultSize);
  return {
    turretEId,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, Tank);
      turretEId[eid] = 0;
    },
    setTurretEid(eid: number, turretEid: number) {
      turretEId[eid] = turretEid;
    },
  };
});
