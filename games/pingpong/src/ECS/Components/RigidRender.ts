import { createCircleCollider, createRectangleCollider } from '../../Physical/createCollider.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { addComponent, World } from 'bitecs';
import { PhysicalRef } from './Physical.ts';
import { DI } from '../../../../tanks/src/DI';

export function createCirceRR(options: Parameters<typeof createCircle>[1] & Parameters<typeof createCircleCollider>[1], {
    world,
    physicalWorld,
} = DI) {
    const physicalId = createCircleCollider(physicalWorld, options);
    const renderId = createCircle(world, options);
    connectPhysicalToRender(world, physicalId, renderId);
    return renderId;
}

export function createRectangleRR(options: Parameters<typeof createRectangle>[1] & Parameters<typeof createRectangleCollider>[1], {
    world,
    physicalWorld,
} = DI) {
    const physicalId = createRectangleCollider(physicalWorld, options);
    const renderId = createRectangle(world, options);
    connectPhysicalToRender(world, physicalId, renderId);
    return renderId;
}

function connectPhysicalToRender(renderWorld: World, physicalId: number, renderId: number) {
    addComponent(renderWorld, renderId, PhysicalRef);
    PhysicalRef.id[renderId] = physicalId;
}
