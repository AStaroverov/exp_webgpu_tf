import { createRectangleRR } from './RigidRender.ts';
import { World } from '../../../../../src/ECS/world.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { PhysicalWorld } from '../../index.ts';
import { ActiveEvents } from '@dimforge/rapier2d';
import { TColor, TShadow } from '../../../../../src/ECS/Components/Common.ts';

export function createFiled(physicalWorld: PhysicalWorld, renderWorld: World, canvas: HTMLCanvasElement) {
    const common = {
        physicalWorld,
        renderWorld,
        color: [0, 1, 0, 1] as TColor,
        bodyType: RigidBodyType.Fixed,
        gravityScale: 0,
        density: 1,
        collisionEvent: ActiveEvents.NONE,
        shadow: [0, 0] as TShadow,
        rotation: 0,
    };
    // top
    createRectangleRR({
        ...common,
        x: canvas.width / 2,
        y: 0,
        width: canvas.width,
        height: 10,
    });
    // right
    createRectangleRR({
        ...common,
        x: canvas.width,
        y: canvas.height / 2,
        width: 10,
        height: canvas.height,
    });
    // bottom
    // createRectangleRR({
    //     ...common,
    //     x: canvas.width / 2,
    //     y: canvas.height,
    //     width: canvas.width,
    //     height: 10,
    // });
    // left
    createRectangleRR({
        ...common,
        x: 0,
        y: canvas.height / 2,
        width: 10,
        height: canvas.height,
    });
}
