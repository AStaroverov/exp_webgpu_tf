import { createRigidCircle, createRigidRectangle } from '../../Physical/createRigid.ts';
import { createCircle, createRectangle } from '../../../../../renderer/src/ECS/Entities/Shapes.ts';
import { RigidBodyRef, RigidBodyState } from './Physical.ts';
import { Impulse, TorqueImpulse } from './Impulse.ts';
import { GameDI } from '../../DI/GameDI.ts';

export function createCircleRR(
    options: Parameters<typeof createCircle>[1] & Parameters<typeof createRigidCircle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const renderId = createCircle(world, options);
    const physicalId = createRigidCircle(options);
    RigidBodyRef.addComponent(world, renderId, physicalId);
    RigidBodyState.addComponent(world, renderId);
    Impulse.addComponent(world, renderId);
    TorqueImpulse.addComponent(world, renderId);
    return [renderId, physicalId];
}

export function createRectangleRR(
    options: Parameters<typeof createRectangle>[1] & Parameters<typeof createRigidRectangle>[0],
    { world } = GameDI,
): [id: number, physicalId: number] {
    const renderId = createRectangle(world, options);
    const physicalId = createRigidRectangle(options);
    RigidBodyRef.addComponent(world, renderId, physicalId);
    RigidBodyState.addComponent(world, renderId);
    Impulse.addComponent(world, renderId);
    TorqueImpulse.addComponent(world, renderId);
    return [renderId, physicalId];
}

