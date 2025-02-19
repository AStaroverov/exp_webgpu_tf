import { GlobalTransform, setMatrixRotateZ, setMatrixTranslate } from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { DI } from '../../DI';
import { query } from 'bitecs';

export function createApplyRigidBodyDeltaToLocalTransformSystem({ world, physicalWorld } = DI) {
    return function () {
        const entities = query(world, [GlobalTransform, RigidBodyRef]);

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = RigidBodyRef.id[id];
            const globalMatrix = GlobalTransform.matrix.getBatche(id);
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const rotationAngle = rigidBody.rotation();
            const position = rigidBody.translation();

            setMatrixTranslate(globalMatrix, position.x, position.y);
            setMatrixRotateZ(globalMatrix, rotationAngle);
        }
    };
}

