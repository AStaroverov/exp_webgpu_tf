import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createTurretControllerComponent = defineComponent((TurretController, obs) => {
  const shoot = TypedArray.f32(delegate.defaultSize);
  const rotation = TypedArray.f64(delegate.defaultSize);
  return {
    shoot,
    rotation,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, TurretController);
      shoot[eid] = 0;
      rotation[eid] = 0;
    },
    shouldShoot(eid: number): boolean {
      return shoot[eid] > 0;
    },
    setShooting$: obs((eid: number, v: number) => {
      shoot[eid] = v;
    }),
    setRotation$: obs((eid: number, v: number) => {
      rotation[eid] = v;
    }),
  };
});
