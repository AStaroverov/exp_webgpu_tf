import { initWebGPU } from '../../src/gpu.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { createDrawRopeSystem } from '../../src/ECS/Systems/RopeSystem/createDrawRopeSystem.ts';
import { initPhysicalWorld } from './src';
import {
    createSyncRigidBodyToRenderTransformSystem,
} from './src/ECS/Systems/createSyncRigidBodyToRenderTransformSystem.ts';
import { createCirceRR, createRectangleRR } from './src/ECS/Components/RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';
import { createPlatformControllerSystem } from './src/ECS/Systems/createPlatformControllerSystem.ts';
import { ActiveEvents, EventQueue } from '@dimforge/rapier2d';
import { createFiled } from './src/ECS/Components/RigidRenderField.ts';
import { createWorld } from '../../src/ECS/world.ts';

const world = createWorld();
const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);
const physicalWorld = initPhysicalWorld();

const drawRopeSystem = createDrawRopeSystem(world, device);
const drawShapeSystem = createDrawShapeSystem(world, device);

createFiled(physicalWorld, world, canvas);

for (let i = 0; i < 1000; i++) {
    const y = Math.floor(i / 50) * 6;
    const x = 150 + i * 6 - y * 50;
    createRectangleRR({
        x,
        y: y + 50,
        z: 0,
        width: 5,
        height: 5,
        rotation: 0,
        color: [1, 0, 0, 1],
        shadow: [0, 0],
        bodyType: RigidBodyType.Dynamic,
        gravityScale: 0,
        density: 1,
        collisionEvent: ActiveEvents.NONE,
    });
}

createCirceRR({
    x: 300,
    y: 300,
    z: 0,
    radius: 40,
    color: [0, 1, 0, 1],
    shadow: [0, 0],
    bodyType: RigidBodyType.Dynamic,
    gravityScale: 4,
    density: 100,
    collisionEvent: ActiveEvents.NONE,
});

const platformId = createRectangleRR({
    x: canvas.width / 2,
    y: canvas.height - 50,
    z: 0,
    width: 100,
    height: 10,
    rotation: 0,
    color: [1, 0, 0, 1],
    shadow: [0, 0],
    bodyType: RigidBodyType.KinematicPositionBased,
    gravityScale: 1,
    density: 1000,
    collisionEvent: ActiveEvents.NONE,
});

const syncRigidBodyMatrixSystem = createSyncRigidBodyToRenderTransformSystem(world, physicalWorld);
const platformControllerSystem = createPlatformControllerSystem(physicalWorld, canvas, platformId);

const renderFrame = createFrameTick({
    canvas,
    device,
    context,
    background: [173, 193, 120, 255].map(v => v / 255),
    getPixelRatio: () => window.devicePixelRatio,
}, ({ passEncoder }) => {
    syncRigidBodyMatrixSystem();

    drawRopeSystem(passEncoder);
    drawShapeSystem(passEncoder);
});

const eventQueue = new EventQueue(true);
const physicalFrame = () => {
    platformControllerSystem();
    physicalWorld.step(eventQueue);
    // eventQueue.drainCollisionEvents((handle1, handle2, started) => {
    //     console.log('Collision event:', handle1, handle2, started);
    // });
    //
    // eventQueue.drainContactForceEvents(event => {
    //     let handle1 = event.collider1(); // Handle of the first collider involved in the event.
    //     let handle2 = event.collider2(); // Handle of the second collider involved in the event.
    //     /* Handle the contact force event. */
    //     console.log('Contact force event:', handle1, handle2);
    // });
};

frameTasks.addInterval(() => {
    physicalFrame();
    renderFrame();
}, 1);
