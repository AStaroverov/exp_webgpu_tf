import { world } from '../../src/ECS/world.ts';
import { initWebGPU } from '../../src/gpu.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './src';
import {
    createApplyRigidBodyDeltaToLocalTransformSystem,
} from './src/ECS/Systems/createApplyRigidBodyDeltaToLocalTransformSystem.ts';
import { EventQueue } from '@dimforge/rapier2d';
import { createTankRR } from './src/ECS/Components/Tank.ts';
import { DI } from './src/DI';
import { createTransformSystem } from '../../src/ECS/Systems/createTransformSystem.ts';
import { createUpdatePlayerTankPositionSystem } from './src/ECS/Systems/createUpdatePlayerTankPositionSystem.ts';
import { createSpawnerBulletsSystem } from './src/ECS/Systems/createControllBulletSystem.ts';
import { createRectangleRR } from './src/ECS/Components/RigidRender.ts';
import { RigidBodyType } from '@dimforge/rapier2d/src/dynamics/rigid_body.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);
const physicalWorld = initPhysicalWorld();

DI.canvas = canvas;
DI.world = world;
DI.physicalWorld = physicalWorld;

const tankId = createTankRR({
    x: 100,
    y: 100,
    rotation: Math.PI / 1.5,
    color: [1, 0, 0, 1],
});

const tankId2 = createTankRR({
    x: 100,
    y: 500,
    rotation: Math.PI / 2,
    color: [1, 1, 0, 1],
});

const tankId3 = createTankRR({
    x: 500,
    y: 100,
    rotation: Math.PI / 3,
    color: [1, 0, 1, 1],
});

const tankId4 = createTankRR({
    x: 500,
    y: 500,
    rotation: Math.PI / 4,
    color: [1, 0, 1, 1],
});

for (let i = 0; i < 100; i++) {
    createRectangleRR({
        x: 200 + (i * 11) % 122,
        y: 200 + Math.floor(i / 11) * 11,
        width: 10,
        height: 10,
        rotation: 0,
        color: [1, 0, 1, 1],
        bodyType: RigidBodyType.Dynamic,
        gravityScale: 0,
        mass: 1,
    });
}

const spawnBullets = createSpawnerBulletsSystem(tankId);
const execTransformSystem = createTransformSystem(DI.world);
const updatePlayerTankPositionSystem = createUpdatePlayerTankPositionSystem(tankId);
const applyRigidBodyDeltaToLocalTransformSystem = createApplyRigidBodyDeltaToLocalTransformSystem();

const inputFrame = () => {
    updatePlayerTankPositionSystem();
    spawnBullets();
};

const eventQueue = new EventQueue(true);
const physicalFrame = () => {
    physicalWorld.step(eventQueue);

    applyRigidBodyDeltaToLocalTransformSystem();

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

const drawShapeSystem = createDrawShapeSystem(world, device);

const renderFrame = createFrameTick({
    canvas,
    device,
    context,
    background: [173 / 255, 193 / 255, 120 / 255, 1],
}, ({ passEncoder }) => {
    drawShapeSystem(passEncoder);
});


// let timeStart = performance.now();
frameTasks.addInterval(() => {
    // const time = performance.now();
    // const delta = time - timeStart;
    // timeStart = time;
    execTransformSystem();

    physicalFrame();

    renderFrame();

    inputFrame();
}, 1);
