import { GameDI } from '../DI/GameDI.ts';
import { ImpulseJoint } from '@dimforge/rapier2d-simd';

export function removePhysicalJoint(jointPid: number, { physicalWorld } = GameDI): null | ImpulseJoint {
    if (jointPid < 0) return null;
    const joint = physicalWorld.getImpulseJoint(jointPid);
    joint && physicalWorld.removeImpulseJoint(joint, true);
    return joint as unknown as null | ImpulseJoint;
}