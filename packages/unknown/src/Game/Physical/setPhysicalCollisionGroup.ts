import { GameDI } from '../DI/GameDI.ts';
import { CollisionGroup } from './createRigid.ts';
import { getGameComponents } from '../ECS/createGameWorld.ts';

export function setPhysicalCollisionGroup(
    eid: number,
    group: typeof CollisionGroup[keyof typeof CollisionGroup],
    { world, physicalWorld } = GameDI,
) {
    const { RigidBodyRef } = getGameComponents(world);
    const physicalId = RigidBodyRef.id[eid];
    const rigidBody = physicalWorld.getRigidBody(physicalId);

    for (let i = 0; i < rigidBody?.numColliders(); i++) {
        rigidBody.collider(i).setCollisionGroups(group << 16 | group);
    }
}
