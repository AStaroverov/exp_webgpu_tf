import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component, obs } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';

/**
 * JointMotor component for controlling revolute joint motor position.
 * Target position is set during a frame and applied by the joint motor system.
 */
export const JointMotor = component({
    targetPosition: TypedArray.f64(delegate.defaultSize),
    stiffness: TypedArray.f64(delegate.defaultSize),
    damping: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, stiffness: number = 1e6, damping: number = 0.2): void {
        addComponent(world, eid, JointMotor);
        JointMotor.targetPosition[eid] = 0;
        JointMotor.stiffness[eid] = stiffness;
        JointMotor.damping[eid] = damping;
    },

    setTargetPosition$: obs((eid: number, position: number): void => {
        JointMotor.targetPosition[eid] = position;
    }),
});
