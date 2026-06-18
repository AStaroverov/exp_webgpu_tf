import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";
import { VehicleType, EngineType } from "../../Config/index.ts";

export { VehicleType };

export const createVehicleComponent = defineComponent((Vehicle, ctx) => {
  const type = ctx.table.flat(Int8Array);
  const engineType = ctx.table.flat(Int8Array);
  return {
    type,
    engineType,
    addComponent(world: World, eid: EntityId, t: VehicleType) {
      addComponent(world, eid, Vehicle);
      type.set(eid, t);
    },
    setEngineType(eid: number, engine: EngineType) {
      engineType.set(eid, engine);
    },
  };
});
