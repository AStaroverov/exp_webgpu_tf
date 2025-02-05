import {
    CircleColliderOptions,
    createCircleCollider,
    createRectangleCollider,
    RectangleColliderOptions,
} from '../../Physical/createCollider.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { PhysicalWorld } from '../../index.ts';
import { addComponent, IWorld } from 'bitecs';
import { PhysicalRef } from './Physical.ts';

export function createCirceRR(options: CircleColliderOptions & {
    physicalWorld: PhysicalWorld,
    renderWorld: IWorld,
    color: [number, number, number, number],
}) {
    const physicalId = createCircleCollider(options.physicalWorld, options);
    const renderId = createCircle(options.renderWorld, options);
    connectPhysicalToRender(options.renderWorld, physicalId, renderId);
    return renderId;
}

export function createRectangleRR(options: RectangleColliderOptions & {
    physicalWorld: PhysicalWorld,
    renderWorld: IWorld,
    color: [number, number, number, number],
}) {
    const physicalId = createRectangleCollider(options.physicalWorld, options);
    const renderId = createRectangle(options.renderWorld, options);
    connectPhysicalToRender(options.renderWorld, physicalId, renderId);
    return renderId;
}


function connectPhysicalToRender(renderWorld: IWorld, physicalId: number, renderId: number) {
    addComponent(renderWorld, PhysicalRef, renderId);
    PhysicalRef.id[renderId] = physicalId;
}
