import {
    CircleColliderOptions,
    createCircleCollider,
    createRectangleCollider,
    RectangleColliderOptions,
} from '../../Physical/createCollider.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { PhysicalWorld } from '../../index.ts';
import { addComponent, World } from 'bitecs';
import { PhysicalRef } from './Physical.ts';
import { TColor, TShadow } from '../../../../../src/ECS/Components/Common.ts';

export function createCirceRR(options: CircleColliderOptions & {
    physicalWorld: PhysicalWorld,
    renderWorld: World,
    color: TColor,
    shadow: TShadow,
}) {
    const physicalId = createCircleCollider(options.physicalWorld, options);
    const renderId = createCircle(options.renderWorld, options);
    connectPhysicalToRender(options.renderWorld, physicalId, renderId);
    return renderId;
}

export function createRectangleRR(options: RectangleColliderOptions & {
    physicalWorld: PhysicalWorld,
    renderWorld: World,
    color: TColor,
    shadow: TShadow,
}) {
    const physicalId = createRectangleCollider(options.physicalWorld, options);
    const renderId = createRectangle(options.renderWorld, options);
    connectPhysicalToRender(options.renderWorld, physicalId, renderId);
    return renderId;
}

function connectPhysicalToRender(renderWorld: World, physicalId: number, renderId: number) {
    addComponent(renderWorld, renderId, PhysicalRef);
    PhysicalRef.id[renderId] = physicalId;
}
