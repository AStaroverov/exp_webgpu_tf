import { ImpulseJoint } from '@dimforge/rapier2d-simd';
import { PhysicalWorld } from './initPhysicalWorld.ts';

export function removePhysicalJoint(physicalWorld: PhysicalWorld, jointPid: number): null | ImpulseJoint {
    if (jointPid < 0) return null;
    const joint = physicalWorld.getImpulseJoint(jointPid);
    joint && physicalWorld.removeImpulseJoint(joint, true);
    return joint as unknown as null | ImpulseJoint;
}