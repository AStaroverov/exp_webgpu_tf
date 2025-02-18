import { initWebGPU } from '../../src/gpu.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './src';
import {
    createApplyRigidBodyDeltaToLocalTransformSystem,
} from './src/ECS/Systems/createApplyRigidBodyDeltaToLocalTransformSystem.ts';
import { EventQueue } from '@dimforge/rapier2d';
import { DI } from './src/DI';
import { createTransformSystem } from '../../src/ECS/Systems/createTransformSystem.ts';
import {
    createPlayerTankBulletSystem,
    createPlayerTankPositionSystem,
    createPlayerTankTurretRotationSystem,
} from './src/ECS/Systems/playerTankControllerSystems.ts';
import { createSpawnerBulletsSystem } from './src/ECS/Systems/createBulletSystem.ts';
import { stats } from './src/stats.ts';
import { getEntityIdByPhysicalId } from './src/ECS/Components/Physical.ts';
import { createWorld, hasComponent, setDefaultSize } from 'bitecs';
import { hit, Hitable } from './src/ECS/Components/Hitable.ts';
import { createHitableSystem } from './src/ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './src/ECS/Systems/createTankAliveSystem.ts';
import { createTankPositionSystem, createTankTurretRotationSystem } from './src/ECS/Systems/tankControllerSystems.ts';
import { createOutZoneDestroySystem } from './src/ECS/Systems/createOutZoneDestroySystem.ts';
import { createTankInputTensorSystem } from './src/ECS/Systems/createTankInputTensorSystem.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);

DI.canvas = canvas;

export function createGame() {
    setDefaultSize(100000);
    const world = DI.world = createWorld();
    const physicalWorld = DI.physicalWorld = initPhysicalWorld();

    // const updateMap = createMapSystem();
    const spawnBullets = createSpawnerBulletsSystem();
    const execTransformSystem = createTransformSystem(world);

    const updateTankPosition = createTankPositionSystem();
    const updateTankTurretRotation = createTankTurretRotationSystem();

    const updatePlayerBullet = createPlayerTankBulletSystem();
    const updatePlayerTankPosition = createPlayerTankPositionSystem();
    const updatePlayerTankTurretRotation = createPlayerTankTurretRotationSystem();

    const applyRigidBodyDeltaToLocalTransformSystem = createApplyRigidBodyDeltaToLocalTransformSystem();
    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();

    const inputFrame = () => {
        updatePlayerBullet();
        updatePlayerTankPosition();
        updatePlayerTankTurretRotation();
        spawnBullets();
    };

    const eventQueue = new EventQueue(true);
    const physicalFrame = (delta: number) => {
        updateTankPosition(delta);
        updateTankTurretRotation(delta);

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
        background: [173, 193, 120, 255].map(v => v / 255),
        getPixelRatio: () => window.devicePixelRatio,
    }, ({ passEncoder }) => {
        drawShapeSystem(passEncoder);
    });

    const destroyOutZone = createOutZoneDestroySystem();
    const destroyFrame = () => {
        destroyOutZone();
    };

    const updateTankInputTensor = createTankInputTensorSystem();
    const statsFrame = () => {
        updateTankInputTensor();
    };

    DI.gameTick = (delta: number) => {
        execTransformSystem();

        physicalFrame(delta);

        // updateMap();

        stats.begin();
        renderFrame();
        stats.end();
        stats.update();

        destroyFrame();
        statsFrame();

        inputFrame();
    };

    return DI;
}