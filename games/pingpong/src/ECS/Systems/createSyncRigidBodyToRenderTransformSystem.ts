import { PhysicalWorld } from '../../index.ts';
import { GlobalTransform, setMatrixRotateZ, setMatrixTranslate } from '../../../../../src/ECS/Components/Transform.ts';
import { PhysicalRef } from '../Components/Physical.ts';
import { query, World } from 'bitecs';

export function createSyncRigidBodyToRenderTransformSystem(world: World, physicalWorld: PhysicalWorld) {
    return function syncRigidBodyToRenderTransformSystem() {
        const entities = query(world, [GlobalTransform, PhysicalRef]);
        // const translation = vec3.create();

        for (let i = 0; i < entities.length; i++) {
            const id = entities[i];
            const physicalId = PhysicalRef.id[id];
            const globalMatrix = GlobalTransform.matrix.getBatche(id);
            const rigidBody = physicalWorld.getRigidBody(physicalId);
            const rotationAngle = rigidBody.rotation();
            const position = rigidBody.translation();

            setMatrixTranslate(globalMatrix, position.x, position.y);
            setMatrixRotateZ(globalMatrix, rotationAngle);
        }
    };
}