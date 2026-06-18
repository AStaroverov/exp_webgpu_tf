import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createVehicleTurretComponent = defineComponent((VehicleTurret, ctx) => {
  const rotationSpeed = ctx.table.flat(Float32Array);
  return {
    rotationSpeed,
    addComponent(world: World, eid: EntityId, speed: number) {
      addComponent(world, eid, VehicleTurret);
      rotationSpeed.set(eid, speed);
    },
  };
});
