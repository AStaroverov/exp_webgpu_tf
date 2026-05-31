import { CollisionGroup } from './createRigid.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../ECS/createPhysicsWorld.ts';
import { PhysicalWorld } from './initPhysicalWorld.ts';

export function setPhysicalCollisionGroup(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    eid: number,
    group: typeof CollisionGroup[keyof typeof CollisionGroup],
) {
    const { RigidBodyRef } = getPhysicsWorldComponents(world);
    const physicalId = RigidBodyRef.id[eid];
    const rigidBody = physicalWorld.getRigidBody(physicalId);

    for (let i = 0; i < rigidBody?.numColliders(); i++) {
        rigidBody.collider(i).setCollisionGroups(group << 16 | group);
    }
}
