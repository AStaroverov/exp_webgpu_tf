import { query } from 'bitecs';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

export function createRigidBodyStateSystem({ physicsWorld: world, physicalWorld } = Worlds) {
    const { RigidBodyRef, RigidBodyState } = getPhysicsWorldComponents(world);

    return () => {
        const entities = query(world, [RigidBodyRef, RigidBodyState]);

        for (let i = 0; i < entities.length; i++) {
            const eid = entities[i];
            const pid = RigidBodyRef.id[eid];
            const rb = physicalWorld.getRigidBody(pid);
            const translation = rb.translation();
            const rotation = rb.rotation();
            const linvel = rb.linvel();
            const angvel = rb.angvel();

            RigidBodyState.update(eid, translation, rotation, linvel, angvel);
        }
    };
}
