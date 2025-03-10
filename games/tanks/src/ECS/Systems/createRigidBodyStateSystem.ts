import { query } from 'bitecs';
import { DI } from '../../DI';
import { RigidBodyRef, RigidBodyState } from '../Components/Physical.ts';

export function createRigidBodyStateSystem({ world, physicalWorld } = DI) {
    return () => {
        const entities = query(world, [RigidBodyRef, RigidBodyState]);

        for (let i = 0; i < entities.length; i++) {
            const eid = entities[i];
            const pid = RigidBodyRef.id[eid];
            const rb = physicalWorld.getRigidBody(pid);
            const translation = rb.translation();
            const rotation = rb.rotation();
            const linvel = rb.linvel();
            // const angvel = rb.angvel();

            RigidBodyState.update(
                eid,
                translation,
                rotation,
                linvel,
                // angvel,
            );
        }
    };
}