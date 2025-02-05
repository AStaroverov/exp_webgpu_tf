import { PhysicalWorld } from '../../index.ts';
import { defineQuery } from 'bitecs';
import { Transform } from '../../../../../src/ECS/Components/Transform.ts';
import { PhysicalRef } from '../Components/Physical.ts';
import { renderWorld } from '../../../../../src/ECS/renderWorld.ts';
import { mat4, vec3 } from 'gl-matrix';

export function createSyncRigidBodyToRenderTransformSystem(physicalWorld: PhysicalWorld) {
    const query = defineQuery([Transform, PhysicalRef]);

    return function syncRigidBodyToRenderTransformSystem() {
        const entities = query(renderWorld);
        const translation = vec3.create();

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = PhysicalRef.id[id];
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const matrix = Transform.matrix[id];

            const position = rigidBody.translation();
            translation[0] = position.x;
            translation[1] = position.y;
            const rotationAngle = rigidBody.rotation();

            mat4.identity(matrix);
            mat4.translate(matrix, matrix, translation);
            mat4.rotateZ(matrix, matrix, rotationAngle);
        }
    };
}