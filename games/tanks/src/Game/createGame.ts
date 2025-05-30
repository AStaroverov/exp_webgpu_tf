import { initWebGPU } from '../../../../src/gpu.ts';
import { createFrameTick } from '../../../../src/WGSL/createFrame.ts';
import { createDrawShapeSystem } from '../../../../src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { initPhysicalWorld } from './Physical/initPhysicalWorld.ts';
import { createApplyRigidBodyToTransformSystem } from './ECS/Systems/createApplyRigidBodyToTransformSystem.ts';
import { EventQueue } from '@dimforge/rapier2d-simd';
import { GameDI } from './DI/GameDI.ts';
import { createTransformSystem } from '../../../../src/ECS/Systems/TransformSystem.ts';
import {
    createPlayerTankBulletSystem,
    createPlayerTankPositionSystem,
    createPlayerTankTurretRotationSystem,
} from './ECS/Systems/PlayerTankControllerSystems.ts';
import { createSpawnerBulletsSystem } from './ECS/Systems/createBulletSystem.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from './ECS/Components/Physical.ts';
import { createWorld, deleteWorld, hasComponent, resetWorld } from 'bitecs';
import { Hitable } from './ECS/Components/Hitable.ts';
import { createHitableSystem } from './ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './ECS/Systems/Tank/createTankAliveSystem.ts';
import { createTankPositionSystem, createTankTurretRotationSystem } from './ECS/Systems/Tank/TankControllerSystems.ts';
import { createDestroyOutOfZoneSystem } from './ECS/Systems/createDestroyOutOfZoneSystem.ts';
import { destroyChangeDetectorSystem } from '../../../../src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDestroyByTimeoutSystem } from './ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createTankAimSystem } from './ECS/Systems/Tank/createTankAimSystem.ts';
import { createDrawGrassSystem } from './ECS/Systems/Render/Grass/createDrawGrassSystem.ts';
import { createRigidBodyStateSystem } from './ECS/Systems/createRigidBodyStateSystem.ts';
import { createDestroySystem } from './ECS/Systems/createDestroySystem.ts';
import { RenderDI } from './DI/RenderDI.ts';
import { PlayerEnvDI } from './DI/PlayerEnvDI.ts';
import { TenserFlowDI } from './DI/TenserFlowDI.ts';
import { createVisualizationTracksSystem } from './ECS/Systems/Tank/createVisualizationTracksSystem.ts';
import { createPostEffect } from './ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts';
import { createTankDecayOutOfZoneSystem } from './ECS/Systems/Tank/createTankDecayOutOfZoneSystem.ts';
import { GameSession } from './ECS/Entities/GameSession.ts';

export type Game = ReturnType<typeof createGame>;

export function createGame({ width, height, withPlayer }: {
    width: number,
    height: number,
    withPlayer: boolean
}) {
    const world = createWorld();
    const physicalWorld = initPhysicalWorld();

    GameDI.width = width;
    GameDI.height = height;
    GameDI.world = world;
    GameDI.physicalWorld = physicalWorld;

    GameDI.setRenderTarget = async (canvas: null | undefined | HTMLCanvasElement) => {
        if (canvas === RenderDI.canvas) {
            return;
        }

        RenderDI.destroy?.();

        if (canvas == null) {
            return;
        }

        const { device, context } = await initWebGPU(canvas);
        RenderDI.canvas = canvas;
        RenderDI.device = device;
        RenderDI.context = context;

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

        RenderDI.renderFrame = (delta: number) => {
            const commandEncoder = device.createCommandEncoder();
            renderFrame(commandEncoder, delta);
            postEffectFrame(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        };

        RenderDI.destroy = () => {
            RenderDI.canvas = null!;
            RenderDI.device = null!;
            RenderDI.context = null!;
            RenderDI.renderFrame = null!;
        };

        if (withPlayer) {
            PlayerEnvDI.destroy?.();

            PlayerEnvDI.document = document;
            PlayerEnvDI.window = window;

            const updatePlayerBullet = createPlayerTankBulletSystem();
            const updatePlayerTankPosition = createPlayerTankPositionSystem();
            const updatePlayerTankTurretRotation = createPlayerTankTurretRotationSystem();

            PlayerEnvDI.inputFrame = () => {
                updatePlayerBullet.tick();
                updatePlayerTankPosition.tick();
                updatePlayerTankTurretRotation.tick();
            };

            PlayerEnvDI.destroy = () => {
                updatePlayerBullet.destroy();
                updatePlayerTankPosition.destroy();
                updatePlayerTankTurretRotation.destroy();

                PlayerEnvDI.document = null!;
                PlayerEnvDI.window = null!;
                PlayerEnvDI.destroy = null!;
                PlayerEnvDI.inputFrame = null!;
            };
        }
    };

    // const updateMap = createMapSystem();
    const execTransformSystem = createTransformSystem(world);

    const updateTankPosition = createTankPositionSystem();
    const updateTankTurretRotation = createTankTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyRigidBodyDeltaToLocalTransform = createApplyRigidBodyToTransformSystem();

    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();

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
            if (event.totalForceMagnitude() < 5_000_000) return;

            const eid1 = rb1 && getEntityIdByPhysicalId(rb1.handle);
            const eid2 = rb2 && getEntityIdByPhysicalId(rb2.handle);

            if (eid1 == null || eid2 == null) return;

            if (hasComponent(world, eid1, Hitable)) {
                Hitable.hit$(eid1, eid2, 1);
            }
            if (hasComponent(world, eid2, Hitable)) {
                Hitable.hit$(eid2, eid1, 1);
            }
        });

        updateHitableSystem();
        updateTankAliveSystem();
    };

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

    const aimUpdate = createTankAimSystem();
    const visTracksUpdate = createVisualizationTracksSystem();

    GameDI.gameTick = (delta: number) => {
        spawnFrame(delta);

        physicalFrame(delta);

        aimUpdate(delta);
        visTracksUpdate(delta);
        // updateMap();

        // stats.begin();
        RenderDI.renderFrame?.(delta);
        // stats.end();
        // stats.update();

        destroyFrame(delta);

        PlayerEnvDI.inputFrame?.();
    };

    GameDI.destroy = () => {
        physicalWorld.free();
        RigidBodyRef.dispose();

        resetWorld(world);
        deleteWorld(world);
        destroyChangeDetectorSystem(world);

        GameSession.reset();

        GameDI.width = null!;
        GameDI.height = null!;
        GameDI.world = null!;
        GameDI.physicalWorld = null!;
        GameDI.gameTick = null!;
        GameDI.destroy = null!;

        RenderDI.destroy?.();
        PlayerEnvDI.destroy?.();

        TenserFlowDI.enabled = false;
    };

    return GameDI;
}