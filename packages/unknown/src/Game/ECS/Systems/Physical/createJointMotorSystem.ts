import { onSet, query } from 'bitecs';
import { RevoluteImpulseJoint } from '@dimforge/rapier2d-simd';
import { createChangeDetector } from '../../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { Worlds } from '../../../DI/Worlds.ts';

export function createJointMotorSystem({ physicsWorld: world, physicalWorld } = Worlds) {
    const { Joint, JointMotor } = getPhysicsWorldComponents(world);
    const motorChanges = createChangeDetector(world, [onSet(JointMotor)]);

    return (_delta: number) => {
        if (!motorChanges.hasChanges()) return;

        const motorEids = query(world, [Joint, JointMotor]);

        for (let i = 0; i < motorEids.length; i++) {
            const eid = motorEids[i];

            if (!motorChanges.has(eid)) continue;

            const jointPid = Joint.pid[eid];
            const joint = physicalWorld.getImpulseJoint(jointPid) as RevoluteImpulseJoint;

            if (joint) {
                joint.configureMotorPosition(
                    JointMotor.targetPosition[eid],
                    JointMotor.stiffness[eid],
                    JointMotor.damping[eid],
                );
            }
        }

        motorChanges.clear();
    };
}
