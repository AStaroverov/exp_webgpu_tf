import { defineQuery } from 'bitecs';
import {
    getMatrixRotationZ,
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
    LocalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { RigidBodyRef } from '../Components/Physical.ts';
import { mat4, vec3 } from 'gl-matrix';
import { DI } from '../../DI';
import { EPSILON } from '../../consts.ts';

export function createApplyRigidBodyDeltaToLocalTransformSystem({ world, physicalWorld } = DI) {
    const query = defineQuery([LocalTransform, GlobalTransform, RigidBodyRef]);
    const translation = vec3.create();

    return function applyRigidBodyDeltaToLocalTransformSystem() {
        const entities = query(world);

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = RigidBodyRef.id[id];
            const localMatrix = LocalTransform.matrix[id];
            const globalMatrix = GlobalTransform.matrix[id];
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const rotationAngle = rigidBody.rotation();
            const position = rigidBody.translation();
            const dX = getMatrixTranslationX(globalMatrix) - position.x;
            const dY = getMatrixTranslationY(globalMatrix) - position.y;
            const dR = -getMatrixRotationZ(globalMatrix) - rotationAngle;

            if (EPSILON < Math.abs(dX) || EPSILON < Math.abs(dY) || EPSILON < Math.abs(dR)) {
                translation[0] = dX;
                translation[1] = dY;
                mat4.translate(localMatrix, localMatrix, translation);
            }
            if (EPSILON < Math.abs(dR)) {
                mat4.rotateZ(localMatrix, localMatrix, dR);
            }
        }
    };
}

