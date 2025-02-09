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
import {
    createSyncGlobalTransformToRigidBodySystem,
} from './src/ECS/Systems/createSyncGlobalTransformToRigidBodySystem.ts';
import { createUpdatePlayerTankPositionSystem } from './src/ECS/Systems/createUpdatePlayerTankPositionSystem.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);
const physicalWorld = initPhysicalWorld();

DI.canvas = canvas;
DI.world = world;
DI.physicalWorld = physicalWorld;


const tankId = createTankRR({
    x: 300,
    y: 300,
    scale: 1,
    rotation: Math.PI / 13,
    color: [1, 0, 0, 1],
});


const execTransformSystem = createTransformSystem(DI.world);
const updatePlayerTankPositionSystem = createUpdatePlayerTankPositionSystem(tankId);
const syncGlobalTransformToRigidBodySystem = createSyncGlobalTransformToRigidBodySystem();
const applyRigidBodyDeltaToLocalTransformSystem = createApplyRigidBodyDeltaToLocalTransformSystem();

const inputFrame = () => {
    updatePlayerTankPositionSystem();
};

const eventQueue = new EventQueue(true);
const physicalFrame = () => {

    syncGlobalTransformToRigidBodySystem();

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

const renderFrame = createFrameTick(canvas, device, context, ({ passEncoder }) => {
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
