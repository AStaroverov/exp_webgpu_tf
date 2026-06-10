import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createJointMotorComponent = defineComponent((JointMotor, obs) => {
  const targetPosition = TypedArray.f64(delegate.defaultSize);
  const stiffness = TypedArray.f64(delegate.defaultSize);
  const damping = TypedArray.f64(delegate.defaultSize);

  return {
    targetPosition,
    stiffness,
    damping,
    addComponent(world: World, eid: EntityId, stiff: number = 1e6, damp: number = 0.2) {
      addComponent(world, eid, JointMotor);
      targetPosition[eid] = 0;
      stiffness[eid] = stiff;
      damping[eid] = damp;
    },
    setTargetPosition$: obs((eid: number, position: number) => {
      targetPosition[eid] = position;
    }),
  };
});
