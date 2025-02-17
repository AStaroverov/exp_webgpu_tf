import { addRigidBodyRef } from './Physical.ts';
import { DI } from '../../DI';
import { addEntity } from 'bitecs';
import { createRigidCircle, createRigidRectangle } from '../../Physical/createRigid.ts';

export function createCircleRigidGroup(
    options: Parameters<typeof createRigidCircle>[0],
    { world } = DI,
): [id: number, physicalId: number] {
    const eid = addEntity(world);
    const physicalId = createRigidCircle(options);
    addRigidBodyRef(eid, physicalId);
    return [eid, physicalId];
}

export function createRectangleRigidGroup(
    options: Parameters<typeof createRigidRectangle>[0],
    { world } = DI,
): [id: number, physicalId: number] {
    const eid = addEntity(world);
    const physicalId = createRigidRectangle(options);
    addRigidBodyRef(eid, physicalId);
    return [eid, physicalId];
}

