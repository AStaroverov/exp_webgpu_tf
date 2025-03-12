import { initWebGPU } from '../../src/gpu.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './src';
import { createApplyRigidBodyToTransformSystem } from './src/ECS/Systems/createApplyRigidBodyToTransformSystem.ts';
import { EventQueue } from '@dimforge/rapier2d';
import { DI } from './src/DI';
import { createTransformSystem } from '../../src/ECS/Systems/TransformSystem.ts';
import {
    createPlayerTankBulletSystem,
    createPlayerTankPositionSystem,
    createPlayerTankTurretRotationSystem,
} from './src/ECS/Systems/playerTankControllerSystems.ts';
import { createSpawnerBulletsSystem } from './src/ECS/Systems/createBulletSystem.ts';
import { getEntityIdByPhysicalId } from './src/ECS/Components/Physical.ts';
import { createWorld, deleteWorld, hasComponent, resetWorld } from 'bitecs';
import { Hitable } from './src/ECS/Components/Hitable.ts';
import { createHitableSystem } from './src/ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './src/ECS/Systems/createTankAliveSystem.ts';
import { createTankPositionSystem, createTankTurretRotationSystem } from './src/ECS/Systems/tankControllerSystems.ts';
import { createOutZoneDestroySystem } from './src/ECS/Systems/createOutZoneDestroySystem.ts';
import { createTankInputTensorSystem } from './src/ECS/Systems/createTankInputTensorSystem.ts';
import {
    createChangeDetectorSystem,
    destroyChangeDetectorSystem,
} from '../../src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDestroyByTimeoutSystem } from './src/ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createAimSystem } from './src/ECS/Systems/createAimSystem.ts';
import { createPostEffect } from './src/ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts';
import { createDrawGrassSystem } from './src/ECS/Systems/Render/Grass/createDrawGrassSystem.ts';
import { createRigidBodyStateSystem } from './src/ECS/Systems/createRigidBodyStateSystem.ts';
import { createDestroySystem } from './src/ECS/Systems/createDestroySystem.ts';

const canvas = document.querySelector('canvas')!;
const { device, context } = await initWebGPU(canvas);

DI.canvas = canvas;

export function createGame() {
    const world = DI.world = createWorld();
    const physicalWorld = DI.physicalWorld = initPhysicalWorld();
    const updateChangedDetector = createChangeDetectorSystem(world);

    // const updateMap = createMapSystem();
    const execTransformSystem = createTransformSystem(world);

    const updateTankPosition = createTankPositionSystem();
    const updateTankTurretRotation = createTankTurretRotationSystem();

    const updatePlayerBullet = createPlayerTankBulletSystem();
    const updatePlayerTankPosition = createPlayerTankPositionSystem();
    const updatePlayerTankTurretRotation = createPlayerTankTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyRigidBodyDeltaToLocalTransform = createApplyRigidBodyToTransformSystem();

    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();

    const inputFrame = () => {
        updatePlayerBullet();
        updatePlayerTankPosition();
        updatePlayerTankTurretRotation();
    };

    const eventQueue = new EventQueue(true);
    const physicalFrame = (delta: number) => {
        updateTankPosition(delta);
        updateTankTurretRotation(delta);

        execTransformSystem();
        physicalWorld.step(eventQueue);
        syncRigidBodyState();
        applyRigidBodyDeltaToLocalTransform();

        // eventQueue.drainCollisionEvents((handle1, handle2, started) => {
        //     console.log('Collision event:', handle1, handle2, started);
        // });

        eventQueue.drainContactForceEvents(event => {
            let handle1 = event.collider1(); // Handle of the first collider involved in the event.
            let handle2 = event.collider2(); // Handle of the second collider involved in the event.

            const rb1 = physicalWorld.getCollider(handle1).parent();
            const rb2 = physicalWorld.getCollider(handle2).parent();

            // TODO: Replace magic number with a constant.
            if (event.totalForceMagnitude() > 5_000_000) {
                const eid1 = rb1 && getEntityIdByPhysicalId(rb1.handle);
                const eid2 = rb2 && getEntityIdByPhysicalId(rb2.handle);

                if (eid1 && hasComponent(world, eid1, Hitable)) {
                    Hitable.hit$(eid1, 1);
                }
                if (eid2 && hasComponent(world, eid2, Hitable)) {
                    Hitable.hit$(eid2, 1);
                }
            }
        });

        updateHitableSystem();
        updateTankAliveSystem();
    };


    const drawGrass = createDrawGrassSystem(device);
    const drawShape = createDrawShapeSystem(world, device);
    const { renderFrame, renderTexture } = createFrameTick({
        canvas,
        device,
        context,
        background: [173, 193, 120, 255].map(v => v / 255),
        getPixelRatio: () => window.devicePixelRatio,
    }, ({ passEncoder, delta }) => {
        drawGrass(passEncoder, delta);
        drawShape(passEncoder);
    });
    const postEffectFrame = createPostEffect(device, context, renderTexture);

    const spawnBullets = createSpawnerBulletsSystem();
    const spawnFrame = (delta: number) => {
        spawnBullets(delta);
    };

    const destroy = createDestroySystem();
    const destroyOutZone = createOutZoneDestroySystem();
    const destroyByTimeout = createDestroyByTimeoutSystem();
    const destroyFrame = (delta: number) => {
        destroyByTimeout(delta);
        destroyOutZone();
        destroy();
    };

    const updateTankInputTensor = createTankInputTensorSystem();
    const statsFrame = () => {
        updateTankInputTensor();
    };

    const aimUpdate = createAimSystem();


    DI.gameTick = (delta: number, withDraw: boolean = true) => {
        spawnFrame(delta);

        physicalFrame(delta);

        aimUpdate(delta);
        // updateMap();

        // stats.begin();
        if (withDraw) {
            const commandEncoder = device.createCommandEncoder();
            renderFrame(commandEncoder, delta);
            postEffectFrame(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        }
        // stats.end();
        // stats.update();
        //
        destroyFrame(delta);

        statsFrame();

        // inputFrame();
        updateChangedDetector();
    };

    DI.destroy = () => {
        physicalWorld.free();
        resetWorld(world);
        deleteWorld(world);
        destroyChangeDetectorSystem(world);

        DI.world = null!;
        DI.physicalWorld = null!;
        DI.gameTick = null!;
        DI.destroy = null!;
    };

    return DI;
}