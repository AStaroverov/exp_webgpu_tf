import { createRectangleRR } from './RigidRender.ts';
import { World } from '../../../../../src/ECS/world.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { PhysicalWorld } from '../../index.ts';
import { ActiveEvents } from '@dimforge/rapier2d';

export function createFiled(physicalWorld: PhysicalWorld, world: World, canvas: HTMLCanvasElement) {
    const common = {
        physicalWorld,
        world,
        color: [0, 1, 0, 1] as [number, number, number, number],
        bodyType: RigidBodyType.Fixed,
        gravityScale: 0,
        mass: 1,
        collisionEvent: ActiveEvents.NONE,
    };
    // top
    createRectangleRR({
        ...common,
        x: canvas.width / 2,
        y: 0,
        width: canvas.width,
        height: 10,
        rotation: 0,
    });
    // right
    createRectangleRR({
        ...common,
        x: canvas.width,
        y: canvas.height / 2,
        width: 10,
        height: canvas.height,
        rotation: 0,
    });
    // bottom
    // createRectangleRR({
    //     ...common,
    //     x: canvas.width / 2,
    //     y: canvas.height,
    //     width: canvas.width,
    //     height: 10,
    //     rotation: 0,
    // });
    // left
    createRectangleRR({
        ...common,
        x: 0,
        y: canvas.height / 2,
        width: 10,
        height: canvas.height,
        rotation: 0,
    });
}
