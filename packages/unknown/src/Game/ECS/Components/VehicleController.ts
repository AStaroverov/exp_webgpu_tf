import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createVehicleControllerComponent = defineComponent((VehicleController, { obs }) => {
  const move = TypedArray.f64(delegate.defaultSize);
  const rotation = TypedArray.f64(delegate.defaultSize);
  return {
    move,
    rotation,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, VehicleController);
      move[eid] = 0;
      rotation[eid] = 0;
    },
    setMove$: obs((eid: number, dir: number) => {
      move[eid] = dir;
    }),
    setRotate$: obs((eid: number, dir: number) => {
      rotation[eid] = dir;
    }),
  };
});
