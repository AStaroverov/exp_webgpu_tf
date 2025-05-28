import { GameDI } from '../DI/GameDI.ts';
import { RigidBodyRef } from '../ECS/Components/Physical.ts';

export function changePhysicalDensity(pid: number, density: number, { physicalWorld } = GameDI) {
    const physicalId = RigidBodyRef.id[pid];
    const rigidBody = physicalWorld.getRigidBody(physicalId);

    if (rigidBody == null) return;

    for (let i = 0; i < rigidBody.numColliders(); i++) {
        rigidBody.collider(i).setDensity(density);
    }
}