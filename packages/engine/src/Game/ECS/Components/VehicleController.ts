import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createVehicleControllerComponent = defineComponent(
  (VehicleController, { obs, table }) => {
    const move = table.flat(Float64Array);
    const rotation = table.flat(Float64Array);
    return {
      move,
      rotation,
      addComponent(world: World, eid: number) {
        addComponent(world, eid, VehicleController);
      },
      setMove$: obs((eid: number, dir: number) => {
        move.set(eid, dir);
      }),
      setRotate$: obs((eid: number, dir: number) => {
        rotation.set(eid, dir);
      }),
    };
  },
);
