import { addEntity } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { createRigidCircle, createRigidRectangle } from '../../Physical/createRigid.ts';
import { getGameComponents } from '../createGameWorld.ts';

export function createCircleRigidGroup(
    options: Parameters<typeof createRigidCircle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const { RigidBodyRef, RigidBodyState, Impulse, TorqueImpulse } = getGameComponents(world);
    const eid = addEntity(world);
    const physicalId = createRigidCircle(options);
    RigidBodyRef.addComponent(world, eid, physicalId);
    RigidBodyState.addComponent(world, eid);
    Impulse.addComponent(world, eid);
    TorqueImpulse.addComponent(world, eid);
    return [eid, physicalId];
}

export function createRectangleRigidGroup(
    options: Parameters<typeof createRigidRectangle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const { RigidBodyRef, RigidBodyState, Impulse, TorqueImpulse } = getGameComponents(world);
    const eid = addEntity(world);
    const physicalId = createRigidRectangle(options);
    RigidBodyRef.addComponent(world, eid, physicalId);
    RigidBodyState.addComponent(world, eid);
    Impulse.addComponent(world, eid);
    TorqueImpulse.addComponent(world, eid);
    return [eid, physicalId];
}
