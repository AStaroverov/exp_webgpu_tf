import { defineQuery } from 'bitecs';
import {
    GlobalTransform,
    LocalTransform,
    setMatrixRotateZ,
    setMatrixTranslate,
} from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { DI } from '../../DI';

export function createApplyRigidBodyDeltaToLocalTransformSystem({ world, physicalWorld } = DI) {
    const query = defineQuery([LocalTransform, GlobalTransform, RigidBodyRef]);

    return function () {
        const entities = query(world);

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = RigidBodyRef.id[id];
            const globalMatrix = GlobalTransform.matrix[id];
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const rotationAngle = rigidBody.rotation();
            const position = rigidBody.translation();

            setMatrixTranslate(globalMatrix, position.x, position.y);
            setMatrixRotateZ(globalMatrix, rotationAngle);
        }
    };
}

