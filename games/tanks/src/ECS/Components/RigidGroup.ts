import { RigidBodyRef, RigidBodyState } from './Physical.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { addEntity } from 'bitecs';
import { createRigidCircle, createRigidRectangle } from '../../Physical/createRigid.ts';

export function createCircleRigidGroup(
    options: Parameters<typeof createRigidCircle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const eid = addEntity(world);
    const physicalId = createRigidCircle(options);
    RigidBodyRef.addComponent(world, eid, physicalId);
    RigidBodyState.addComponent(world, eid);
    return [eid, physicalId];
}

export function createRectangleRigidGroup(
    options: Parameters<typeof createRigidRectangle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const eid = addEntity(world);
    const physicalId = createRigidRectangle(options);
    RigidBodyRef.addComponent(world, eid, physicalId);
    RigidBodyState.addComponent(world, eid);
    return [eid, physicalId];
}

