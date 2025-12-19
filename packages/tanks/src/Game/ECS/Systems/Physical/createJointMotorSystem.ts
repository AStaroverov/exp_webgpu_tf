import { onSet, query } from 'bitecs';
import { RevoluteImpulseJoint } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { Joint } from '../../Components/Joint.ts';
import { JointMotor } from '../../Components/JointMotor.ts';
import { createChangeDetector } from '../../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';

/**
 * System that applies JointMotor target positions to physics joints.
 * Should run after systems that set motor positions and before physics step.
 */
export function createJointMotorSystem({ world, physicalWorld } = GameDI) {
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
