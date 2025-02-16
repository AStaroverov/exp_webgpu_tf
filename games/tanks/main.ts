import { world } from '../../src/ECS/world.ts';
import { initWebGPU } from '../../src/gpu.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './src';
import {
    createApplyRigidBodyDeltaToLocalTransformSystem,
} from './src/ECS/Systems/createApplyRigidBodyDeltaToLocalTransformSystem.ts';
import { Cuboid, EventQueue } from '@dimforge/rapier2d';
import { createTankRR } from './src/ECS/Components/Tank.ts';
import { DI } from './src/DI';
import { createTransformSystem } from '../../src/ECS/Systems/createTransformSystem.ts';
import { createUpdatePlayerTankPositionSystem } from './src/ECS/Systems/createUpdatePlayerTankPositionSystem.ts';
import { createSpawnerBulletsSystem } from './src/ECS/Systems/createControllBulletSystem.ts';
import { stats } from './src/stats.ts';
import { getEntityIdByPhysicalId } from './src/ECS/Components/Physical.ts';
import { hasComponent } from 'bitecs';
import { hit, Hitable } from './src/ECS/Components/Hitable.ts';
import { createHitableSystem } from './src/ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './src/ECS/Systems/createTankAliveSystem.ts';
import { fillEnvironment } from './src/TilesMatrix/fillers/environment.ts';
import { Matrix } from '../../lib/Matrix';
import { getEmptyTile, TileType } from './src/TilesMatrix/def.ts';
import { fillWalls } from './src/TilesMatrix/fillers/walls.ts';
import { createWallRR } from './src/ECS/Components/Wall.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);
const physicalWorld = initPhysicalWorld();

DI.canvas = canvas;
DI.world = world;
DI.physicalWorld = physicalWorld;

const tankId = createTankRR({
    x: 100,
    y: 100,
    rotation: Math.PI / 1.3,
    color: [1, 0, 0, 1],
});
//
// const tankId2 = createTankRR({
//     x: 250,
//     y: 250,
//     rotation: Math.PI / 2,
//     color: [1, 1, 0, 1],
// });
//
// const tankId3 = createTankRR({
//     x: 500,
//     y: 100,
//     rotation: Math.PI / 3,
//     color: [1, 0, 1, 1],
// });

const tankId4 = createTankRR({
    x: 500,
    y: 500,
    rotation: Math.PI / 4,
    color: [1, 0, 1, 1],
});
//
// for (let i = 0; i < 100; i++) {
//     createRectangleRR({
//         x: 200 + (i * 11) % 122,
//         y: 200 + Math.floor(i / 11) * 11,
//         width: 10,
//         height: 10,
//         rotation: 0,
//         color: [1, 0, 1, 1],
//         bodyType: RigidBodyType.Dynamic,
//         gravityScale: 0,
//         mass: 1,
//     });
//
// }
// createRectangleRR({
//     x: 400,
//     y: 400,
//     width: 50,
//     height: 50,
//     rotation: Math.PI / 4,
//     color: [1, 0, 1, 1],
//     bodyType: RigidBodyType.Dynamic,
//     gravityScale: 0,
//     mass: 100,
// });

const matrix = Matrix.create(100, 100, getEmptyTile);
fillEnvironment(matrix);
fillWalls(matrix);

// place tanks
physicalWorld.step();

Matrix.forEach(matrix, (item, x, y) => {
    if (item.type === TileType.wall) {
        const options = {
            x: x * 10,
            y: y * 10,
            width: 10,
            height: 10,
        };
        const intersected = null !== physicalWorld.intersectionWithShape(
            options, 0, new Cuboid(10 * options.width / 2, 10 * options.height / 2),
        );
        !intersected && createWallRR(options);
    }
});

const spawnBullets = createSpawnerBulletsSystem(tankId);
const execTransformSystem = createTransformSystem(DI.world);
const updatePlayerTankPositionSystem = createUpdatePlayerTankPositionSystem(tankId);
const applyRigidBodyDeltaToLocalTransformSystem = createApplyRigidBodyDeltaToLocalTransformSystem();
const updateHitableSystem = createHitableSystem();
const updateTankAliveSystem = createTankAliveSystem();

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

    eventQueue.drainContactForceEvents(event => {
        let handle1 = event.collider1(); // Handle of the first collider involved in the event.
        let handle2 = event.collider2(); // Handle of the second collider involved in the event.

        const rb1 = physicalWorld.getCollider(handle1).parent();
        const rb2 = physicalWorld.getCollider(handle2).parent();

        // TODO: Replace magic number with a constant.
        if (event.totalForceMagnitude() > 2642367.5) {
            const eid1 = rb1 && getEntityIdByPhysicalId(rb1.handle);
            const eid2 = rb2 && getEntityIdByPhysicalId(rb2.handle);

            if (eid1 && hasComponent(world, Hitable, eid1)) {
                hit(eid1, 1);
            }
            if (eid2 && hasComponent(world, Hitable, eid2)) {
                hit(eid2, 1);
            }
        }
    });

    updateHitableSystem();
    updateTankAliveSystem();
};

const drawShapeSystem = createDrawShapeSystem(world, device);

const renderFrame = createFrameTick({
    canvas,
    device,
    context,
    background: [173 / 255, 193 / 255, 120 / 255, 1],
    getPixelRatio: () => window.devicePixelRatio,
}, ({ passEncoder }) => {
    drawShapeSystem(passEncoder);
});

document.body.appendChild(stats.dom);
// let timeStart = performance.now();
frameTasks.addInterval(() => {
    // const time = performance.now();
    // const delta = time - timeStart;
    // timeStart = time;
    execTransformSystem();

    physicalFrame();

    stats.begin();
    renderFrame();
    stats.end();
    stats.update();

    inputFrame();
}, 1);
