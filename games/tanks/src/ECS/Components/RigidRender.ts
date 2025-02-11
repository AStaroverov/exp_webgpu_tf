import {
    createRigidCircle,
    createRigidRectangle,
    RigidCircleOptions,
    RigidRectangleOptions,
} from '../../Physical/createRigid.ts';
import { createCircle, createRectangle } from '../../../../../src/ECS/Entities/Shapes.ts';
import { addRigidBodyRef } from './Physical.ts';
import { DI } from '../../DI';

export function createCirceRR(
    options: RigidCircleOptions & { color: [number, number, number, number], },
    { world } = DI,
): [id: number, physicalId: number] {
    const renderId = createCircle(world, options);
    const physicalId = createRigidCircle(options);
    addRigidBodyRef(world, renderId, physicalId);
    return [renderId, physicalId];
}

export function createRectangleRR(
    options: RigidRectangleOptions & { color: [number, number, number, number], },
    { world } = DI,
): [id: number, physicalId: number] {
    const renderId = createRectangle(world, options);
    const physicalId = createRigidRectangle(options);
    addRigidBodyRef(world, renderId, physicalId);
    return [renderId, physicalId];
}

