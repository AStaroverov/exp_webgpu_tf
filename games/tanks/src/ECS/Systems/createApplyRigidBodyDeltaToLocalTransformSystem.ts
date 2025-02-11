import { defineQuery } from 'bitecs';
import {
    GlobalTransform,
    LocalTransform,
    setMatrixRotateZ,
    setMatrixTranslate,
} from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { DI } from '../../DI';

export const EPSILON = 0.000001;

export function createApplyRigidBodyDeltaToLocalTransformSystem({ world, physicalWorld } = DI) {
    const query = defineQuery([LocalTransform, GlobalTransform, RigidBodyRef]);

    return function () {
        const entities = query(world);

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = RigidBodyRef.id[id];
            // const localMatrix = LocalTransform.matrix[id];
            const globalMatrix = GlobalTransform.matrix[id];
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const rotationAngle = rigidBody.rotation();
            const position = rigidBody.translation();
            // const dX = getMatrixTranslationX(globalMatrix) - position.x;
            // const dY = getMatrixTranslationY(globalMatrix) - position.y;
            // const dR = getMatrixRotationZ(globalMatrix) - rotationAngle;

            setMatrixTranslate(globalMatrix, position.x, position.y);
            setMatrixRotateZ(globalMatrix, rotationAngle);
            // if (Math.abs(dX) > EPSILON || Math.abs(dY) > EPSILON) {
            //     translation[0] = dX;
            //     translation[1] = dY;
            //     debugger
            //     mat4.translate(localMatrix, localMatrix, translation);]
            // }
            // if (Math.abs(dR) > EPSILON) {
            //     debugger
            //     mat4.rotateZ(localMatrix, localMatrix, dR);
            // }
        }
    };
}

