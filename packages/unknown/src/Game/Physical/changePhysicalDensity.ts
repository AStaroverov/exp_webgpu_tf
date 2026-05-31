import { getPhysicsWorldComponents, PhysicsWorld } from '../ECS/createPhysicsWorld.ts';
import { PhysicalWorld } from './initPhysicalWorld.ts';

export function changePhysicalDensity(world: PhysicsWorld, physicalWorld: PhysicalWorld, pid: number, density: number) {
    const { RigidBodyRef } = getPhysicsWorldComponents(world);
    const physicalId = RigidBodyRef.id[pid];
    const rigidBody = physicalWorld.getRigidBody(physicalId);

    if (rigidBody == null) return;

    for (let i = 0; i < rigidBody.numColliders(); i++) {
        rigidBody.collider(i).setDensity(density);
    }
}
