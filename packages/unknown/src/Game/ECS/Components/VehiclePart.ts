import { addComponent, removeComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createVehiclePartComponent = defineComponent((VehiclePart) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, VehiclePart);
  },
  removeComponent(world: World, eid: EntityId) {
    removeComponent(world, eid, VehiclePart);
  },
}));

export const createVehiclePartCaterpillarComponent = defineComponent((VehiclePartCaterpillar) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, VehiclePartCaterpillar);
  },
  removeComponent(world: World, eid: EntityId) {
    removeComponent(world, eid, VehiclePartCaterpillar);
  },
}));
