import { GameDI } from '../DI/GameDI.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';
import { CollisionGroup } from './createRigid.ts';

export function resetCollisionsTo(eid: number, group: CollisionGroup, { physicalWorld } = GameDI) {
    const physicalId = RigidBodyRef.id[eid];
    const rigidBody = physicalWorld.getRigidBody(physicalId);

    for (let i = 0; i < rigidBody?.numColliders(); i++) {
        rigidBody.collider(i).setCollisionGroups(group << 16 | group);
    }
}