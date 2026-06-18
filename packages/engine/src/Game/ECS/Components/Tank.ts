import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createTankComponent = defineComponent((Tank, ctx) => {
  const turretEId = ctx.table.flat(Float64Array);
  return {
    turretEId,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, Tank);
    },
    setTurretEid(eid: number, turretEid: number) {
      turretEId.set(eid, turretEid);
    },
  };
});
