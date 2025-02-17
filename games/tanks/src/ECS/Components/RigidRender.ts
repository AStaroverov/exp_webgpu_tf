import { createRigidCircle, createRigidRectangle } from '../../Physical/createRigid.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { addRigidBodyRef } from './Physical.ts';
import { DI } from '../../DI';

export function createCircleRR(
    options: Parameters<typeof createCircle>[1] & Parameters<typeof createRigidCircle>[0],
    { world } = DI,
): [id: number, physicalId: number] {
    const renderId = createCircle(world, options);
    const physicalId = createRigidCircle(options);
    addRigidBodyRef(world, renderId, physicalId);
    return [renderId, physicalId];
}

export function createRectangleRR(
    options: Parameters<typeof createRectangle>[1] & Parameters<typeof createRigidRectangle>[0],
    { world } = DI,
): [id: number, physicalId: number] {
    const renderId = createRectangle(world, options);
    const physicalId = createRigidRectangle(options);
    addRigidBodyRef(world, renderId, physicalId);
    return [renderId, physicalId];
}

