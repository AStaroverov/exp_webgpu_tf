import { DI } from '../../DI';
import { defineQuery } from 'bitecs';
import { GlobalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { setMatrixToRigidBody } from '../../Physical/setMatrixToRigidBody.ts';

export function createSyncGlobalTransformToRigidBodySystem({ world, physicalWorld } = DI) {
    const query = defineQuery([GlobalTransform, RigidBodyRef]);

    return function execMainTransformSystem() {
        const entities = query(world);

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const matrix = GlobalTransform.matrix[id];
            const rbId = RigidBodyRef.id[id];
            const rigidBody = physicalWorld.getRigidBody(rbId);

            setMatrixToRigidBody(rigidBody, matrix);
        }
    };
}