import { initWebGPU } from '../../src/gpu.ts';
import { createFrameTick } from '../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './src';
import { createApplyRigidBodyToTransformSystem } from './src/ECS/Systems/createApplyRigidBodyToTransformSystem.ts';
import { EventQueue } from '@dimforge/rapier2d-simd';
import { GameDI } from './src/DI/GameDI.ts';
import { createTransformSystem } from '../../src/ECS/Systems/TransformSystem.ts';
import {
    createPlayerTankBulletSystem,
    createPlayerTankPositionSystem,
    createPlayerTankTurretRotationSystem,
} from './src/ECS/Systems/PlayerTankControllerSystems.ts';
import { createSpawnerBulletsSystem } from './src/ECS/Systems/createBulletSystem.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from './src/ECS/Components/Physical.ts';
import { createWorld, deleteWorld, hasComponent, resetWorld } from 'bitecs';
import { Hitable } from './src/ECS/Components/Hitable.ts';
import { createHitableSystem } from './src/ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './src/ECS/Systems/Tank/createTankAliveSystem.ts';
import {
    createTankPositionSystem,
    createTankTurretRotationSystem,
} from './src/ECS/Systems/Tank/TankControllerSystems.ts';
import { createDestroyOutOfZoneSystem } from './src/ECS/Systems/createDestroyOutOfZoneSystem.ts';
import { createTankInputTensorSystem } from './src/ECS/Systems/RL/createTankInputTensorSystem.ts';
import { destroyChangeDetectorSystem } from '../../src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDestroyByTimeoutSystem } from './src/ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createTankAimSystem } from './src/ECS/Systems/Tank/createTankAimSystem.ts';
import { createDrawGrassSystem } from './src/ECS/Systems/Render/Grass/createDrawGrassSystem.ts';
import { createRigidBodyStateSystem } from './src/ECS/Systems/createRigidBodyStateSystem.ts';
import { createDestroySystem } from './src/ECS/Systems/createDestroySystem.ts';
import { RenderDI } from './src/DI/RenderDI.ts';
import { noop } from 'lodash-es';
import { PlayerEnvDI } from './src/DI/PlayerEnvDI.ts';
import { TenserFlowDI } from './src/DI/TenserFlowDI.ts';
import { createVisualizationTracksSystem } from './src/ECS/Systems/Tank/createVisualizationTracksSystem.ts';
import { createPostEffect } from './src/ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts';
import { createTankDecayOutOfZoneSystem } from './src/ECS/Systems/Tank/createTankDecayOutOfZoneSystem.ts';

export async function createGame({ width, height, withRender, withPlayer }: {
    width: number,
    height: number,
    withRender: boolean
    withPlayer: boolean
}) {
    const world = createWorld();
    const physicalWorld = initPhysicalWorld();

    GameDI.width = width;
    GameDI.height = height;
    GameDI.world = world;
    GameDI.physicalWorld = physicalWorld;

    if (withRender && RenderDI.canvas == null) {
        const canvas = document.querySelector('canvas')!;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const { device, context } = await initWebGPU(canvas);
        RenderDI.canvas = canvas;
        RenderDI.device = device;
        RenderDI.context = context;
    }

    if (withPlayer) {
        PlayerEnvDI.container = RenderDI.canvas;
        PlayerEnvDI.document = document;
        PlayerEnvDI.window = window;
    }

    // const updateMap = createMapSystem();
    const execTransformSystem = createTransformSystem(world);

    const updateTankPosition = createTankPositionSystem();
    const updateTankTurretRotation = createTankTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyRigidBodyDeltaToLocalTransform = createApplyRigidBodyToTransformSystem();

    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();

    const inputFrame = withPlayer ? (() => {
        const updatePlayerBullet = createPlayerTankBulletSystem();
        const updatePlayerTankPosition = createPlayerTankPositionSystem();
        const updatePlayerTankTurretRotation = createPlayerTankTurretRotationSystem();

        return () => {
            updatePlayerBullet();
            updatePlayerTankPosition();
            updatePlayerTankTurretRotation();
        };
    })() : noop;

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

    const renderFrame = withRender ? (() => {
        const { canvas, device, context } = RenderDI;
        const drawGrass = createDrawGrassSystem();
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

        return (delta: number) => {
            const commandEncoder = device.createCommandEncoder();
            renderFrame(commandEncoder, delta);
            postEffectFrame(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        };
    })() : noop;


    const spawnBullets = createSpawnerBulletsSystem();
    const spawnFrame = (delta: number) => {
        spawnBullets(delta);
    };

    const destroy = createDestroySystem();
    const destroyOutOfZone = createDestroyOutOfZoneSystem();
    const destroyByTimeout = createDestroyByTimeoutSystem();
    const decayTankOnOutOfZone = createTankDecayOutOfZoneSystem();

    const destroyFrame = (delta: number) => {
        decayTankOnOutOfZone();
        destroyByTimeout(delta);
        destroyOutOfZone();
        destroy();
    };

    const updateTankInputTensor = createTankInputTensorSystem();
    const statsFrame = () => {
        updateTankInputTensor();
    };

    const aimUpdate = createTankAimSystem();
    const visTracksUpdate = createVisualizationTracksSystem();

    GameDI.gameTick = (delta: number) => {
        spawnFrame(delta);

        physicalFrame(delta);

        aimUpdate(delta);
        visTracksUpdate(delta);
        // updateMap();

        // stats.begin();
        renderFrame(delta);
        // stats.end();
        // stats.update();
        //
        destroyFrame(delta);

        statsFrame();

        inputFrame();
    };

    GameDI.destroy = () => {
        physicalWorld.free();
        RigidBodyRef.dispose();

        resetWorld(world);
        deleteWorld(world);
        destroyChangeDetectorSystem(world);

        GameDI.width = null!;
        GameDI.height = null!;
        GameDI.world = null!;
        GameDI.physicalWorld = null!;
        GameDI.gameTick = null!;
        GameDI.destroy = null!;

        PlayerEnvDI.window = null!;
        PlayerEnvDI.document = null!;
        PlayerEnvDI.container = null!;

        TenserFlowDI.enabled = false;
        TenserFlowDI.shouldCollectState = false;
    };

    return GameDI;
}