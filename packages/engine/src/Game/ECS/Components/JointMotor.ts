import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

export const createJointMotorComponent = defineComponent((JointMotor, ctx) => {
  const targetPosition = ctx.table.flat(Float64Array);
  const stiffness = ctx.table.flat(Float64Array);
  const damping = ctx.table.flat(Float64Array);

  return {
    targetPosition,
    stiffness,
    damping,
    addComponent(world: World, eid: EntityId, stiff: number = 1e6, damp: number = 0.2) {
      addComponent(world, eid, JointMotor);
      stiffness.set(eid, stiff);
      damping.set(eid, damp);
    },
    setTargetPosition$: ctx.obs((eid: number, position: number) => {
      targetPosition.set(eid, position);
    }),
  };
});
