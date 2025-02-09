import {
    CircleColliderOptions,
    createCircleCollider,
    createRectangleCollider,
    RectangleColliderOptions,
} from '../../Physical/createCollider.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { World } from '../../../../../src/ECS/world.ts';
import { addComponent } from 'bitecs';
import { RigidBodyRef } from './Physical.ts';
import { DI } from '../../DI';

export function createCirceRR(
    options: CircleColliderOptions & { color: [number, number, number, number], },
    { world } = DI,
) {
    const renderId = createCircle(world, options);
    const physicalId = createCircleCollider(options);
    connectPhysicalToRender(world, physicalId, renderId);
    return renderId;
}

export function createRectangleRR(
    options: RectangleColliderOptions & { color: [number, number, number, number], },
    { world } = DI,
) {
    const renderId = createRectangle(world, options);
    const physicalId = createRectangleCollider(options);
    connectPhysicalToRender(world, physicalId, renderId);
    return renderId;
}


function connectPhysicalToRender(world: World, physicalId: number, renderId: number) {
    addComponent(world, RigidBodyRef, renderId);
    RigidBodyRef.id[renderId] = physicalId;
}
