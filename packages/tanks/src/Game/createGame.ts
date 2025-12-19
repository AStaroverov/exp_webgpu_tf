import { EventQueue } from '@dimforge/rapier2d-simd';
import { createWorld, deleteWorld, EntityId, hasComponent, resetWorld } from 'bitecs';
import { destroyChangeDetectorSystem } from '../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDrawShapeSystem } from '../../../renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { createTransformSystem } from '../../../renderer/src/ECS/Systems/TransformSystem.ts';
import { initWebGPU } from '../../../renderer/src/gpu.ts';
import { createFrameTick } from '../../../renderer/src/WGSL/createFrame.ts';
import { GameDI } from './DI/GameDI.ts';
import { PlayerEnvDI } from './DI/PlayerEnvDI.ts';
import { RenderDI } from './DI/RenderDI.ts';
import { Hitable } from './ECS/Components/Hitable.ts';
import { getEntityIdByPhysicalId, RigidBodyRef } from './ECS/Components/Physical.ts';
import { GameSession } from './ECS/Entities/GameSession.ts';
import { GameMap } from './ECS/Entities/GameMap.ts';
import { SystemGroup } from './ECS/Plugins/systems.ts';
import { createApplyRigidBodyToTransformSystem } from './ECS/Systems/createApplyRigidBodyToTransformSystem.ts';
import { createSpawnerBulletsSystem } from './ECS/Systems/createBulletSystem.ts';
import { createDestroyByTimeoutSystem } from './ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createDestroyBySpeedSystem } from './ECS/Systems/createDestroyBySpeedSystem.ts';
import { createDestroyOutOfZoneSystem } from './ECS/Systems/createDestroyOutOfZoneSystem.ts';
import { createDestroySystem } from './ECS/Systems/createDestroySystem.ts';
import { createRigidBodyStateSystem } from './ECS/Systems/createRigidBodyStateSystem.ts';
import { createApplyImpulseSystem } from './ECS/Systems/createApplyImpulseSystem.ts';
import {
    createPlayerTankBulletSystem,
    createPlayerTankPositionSystem,
    createPlayerTankTurretRotationSystem,
} from './ECS/Systems/PlayerTankControllerSystems.ts';
import { createDrawGrassSystem } from './ECS/Systems/Render/Grass/createDrawGrassSystem.ts';
import { createDrawMuzzleFlashSystem } from './ECS/Systems/Render/MuzzleFlash/createDrawMuzzleFlashSystem.ts';
import { createDrawHitFlashSystem } from './ECS/Systems/Render/HitFlash/createDrawHitFlashSystem.ts';
import { createDrawExplosionSystem } from './ECS/Systems/Render/Explosion/createDrawExplosionSystem.ts';
import { createDrawExhaustSmokeSystem } from './ECS/Systems/Render/ExhaustSmoke/createDrawExhaustSmokeSystem.ts';
import { createPostEffect } from './ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts';
import { createTankDecayOutOfZoneSystem } from './ECS/Systems/Tank/createTankDecayOutOfZoneSystem.ts';
import { createVisualizationTracksSystem } from './ECS/Systems/Tank/createVisualizationTracksSystem.ts';
import { createSpawnTreadMarksSystem } from './ECS/Systems/Tank/createSpawnTreadMarksSystem.ts';
import { createUpdateTreadMarksSystem } from './ECS/Systems/Tank/createUpdateTreadMarksSystem.ts';
import { createSpawnWheelTreadMarksSystem } from './ECS/Systems/Vehicle/createSpawnWheelTreadMarksSystem.ts';
import { createExhaustSmokeSpawnSystem } from './ECS/Systems/Vehicle/createExhaustSmokeSpawnSystem.ts';
import { createVehicleTurretRotationSystem } from './ECS/Systems/Vehicle/VehicleControllerSystems.ts';
import { createTrackControlSystem } from './ECS/Systems/Vehicle/TrackControlSystem.ts';
import { createWheelControlSystem } from './ECS/Systems/Vehicle/WheelControlSystem.ts';
import { createJointMotorSystem } from './ECS/Systems/Physical/createJointMotorSystem.ts';
import { initPhysicalWorld } from './Physical/initPhysicalWorld.ts';
import { createProgressSystem } from './ECS/Systems/createProgressSystem.ts';
import { createCameraSystem, setCameraTarget, setInfiniteMapMode, initCameraPosition, CameraState } from './ECS/Systems/Camera/CameraSystem.ts';
import { setCameraPosition } from '../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { createSoundSystem, loadGameSounds, disposeSoundSystem, SoundManager, createTankMoveSoundSystem } from './ECS/Systems/Sound/index.ts';
import { createDebrisCollectorSystem } from './ECS/Systems/createDebrisCollectorSystem.ts';
import { createHitableSystem } from './ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './ECS/Systems/Tank/createTankAliveSystem.ts';
import { createShieldRegenerationSystem } from './ECS/Systems/createShieldRegenerationSystem.ts';
import { SoundDI } from './DI/SoundDI.ts';

export type Game = ReturnType<typeof createGame>;

export function createGame({ width, height }: {
    width: number,
    height: number,
}) {
    const world = createWorld();
    const physicalWorld = initPhysicalWorld();

    GameDI.width = width;
    GameDI.height = height;
    GameDI.world = world;
    GameDI.physicalWorld = physicalWorld;

    // Initialize map offset to center (bounded mode default)
    GameMap.setOffset(width / 2, height / 2);
    initCameraPosition();

    // const updateMap = createMapSystem();
    const execTransformSystem = createTransformSystem(world);

    // Vehicle control systems - tracks for tanks/harvesters, wheels for cars
    const updateTrackControl = createTrackControlSystem();
    const updateWheelControl = createWheelControlSystem();
    const updateVehicleTurretRotation = createVehicleTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyRigidBodyDeltaToLocalTransform = createApplyRigidBodyToTransformSystem();
    const applyImpulses = createApplyImpulseSystem();
    const applyJointMotors = createJointMotorSystem();

    const eventQueue = new EventQueue(true);
    const physicalFrame = (delta: number) => {
        // Update vehicle controls - tracks and wheels
        updateTrackControl(delta);
        updateWheelControl(delta);
        updateVehicleTurretRotation(delta);

        execTransformSystem();
        applyImpulses();
        applyJointMotors(delta);
        physicalWorld.step(eventQueue);
        syncRigidBodyState();
        applyRigidBodyDeltaToLocalTransform();

        // eventQueue.drainCollisionEvents((handle1, handle2, started) => {
        //     console.log('Collision event:', handle1, handle2, started);
        // });

        eventQueue.drainContactForceEvents(event => {
            const handle1 = event.collider1(); // Handle of the first collider involved in the event.
            const handle2 = event.collider2(); // Handle of the second collider involved in the event.
            const rb1 = physicalWorld.getCollider(handle1).parent();
            const rb2 = physicalWorld.getCollider(handle2).parent();
            const eid1 = rb1 && getEntityIdByPhysicalId(rb1.handle);
            const eid2 = rb2 && getEntityIdByPhysicalId(rb2.handle);

            if (eid1 == null || eid2 == null) return;

            const forceMagnitude = event.totalForceMagnitude();

            if (hasComponent(world, eid1, Hitable)) {
                Hitable.hit$(eid1, eid2, forceMagnitude);
            }
            if (hasComponent(world, eid2, Hitable)) {
                Hitable.hit$(eid2, eid1, forceMagnitude);
            }
        });
    };

    const spawnBullets = createSpawnerBulletsSystem();
    const spawnTreadMarks = createSpawnTreadMarksSystem();
    const spawnWheelTreadMarks = createSpawnWheelTreadMarksSystem();
    const spawnExhaustSmoke = createExhaustSmokeSpawnSystem();
    const spawnFrame = (delta: number) => {
        spawnBullets(delta);
        spawnTreadMarks(delta);
        spawnWheelTreadMarks(delta);
        spawnExhaustSmoke(delta);
    };

    const destroy = createDestroySystem();
    const destroyOutOfZone = createDestroyOutOfZoneSystem();
    const destroyByTimeout = createDestroyByTimeoutSystem();
    const destroyBySpeed = createDestroyBySpeedSystem();
    const decayTankOnOutOfZone = createTankDecayOutOfZoneSystem();

    const destroyFrame = (delta: number) => {
        decayTankOnOutOfZone();
        destroyByTimeout(delta);
        destroyBySpeed();
        destroyOutOfZone();
        destroy();
    };

    const visTracksUpdate = createVisualizationTracksSystem();
    const updateTreadMarks = createUpdateTreadMarksSystem();
    const updateProgress = createProgressSystem();
    const updateCamera = createCameraSystem();
    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();
    const collectDebris = createDebrisCollectorSystem();
    const regenerateShields = createShieldRegenerationSystem();

    GameDI.gameTick = (delta: number) => {
        if (GameDI.world === null) return;

        physicalFrame(delta);

        GameDI.plugins.systems[SystemGroup.Before].forEach(system => system(delta));

        updateHitableSystem(delta);
        updateTankAliveSystem();
        collectDebris(delta);
        regenerateShields(delta);

        visTracksUpdate(delta);
        updateProgress(delta);
        updateTreadMarks();
        // updateMap();

        // Update camera before rendering
        updateCamera(delta);
        setCameraPosition(CameraState.x, CameraState.y);

        destroyFrame(delta);
        spawnFrame(delta);

        PlayerEnvDI.inputFrame?.();
        RenderDI.renderFrame?.(delta);
        SoundDI.soundFrame?.(delta);

        GameDI.plugins.systems[SystemGroup.After].forEach(system => system(delta));
    };

    GameDI.destroy = () => {
        GameDI.plugins.dispose();

        physicalWorld.free();
        RigidBodyRef.dispose();

        resetWorld(world);
        deleteWorld(world);
        destroyChangeDetectorSystem(world);

        GameSession.reset();
        GameMap.reset();

        GameDI.width = null!;
        GameDI.height = null!;
        GameDI.world = null!;
        GameDI.physicalWorld = null!;
        GameDI.gameTick = null!;
        GameDI.destroy = null!;

        SoundDI.destroy?.();
        RenderDI.destroy?.();
        PlayerEnvDI.destroy?.();
    };

    GameDI.enableSound = async () => {
        if (SoundDI.enabled) {
            return;
        }

        SoundDI.enabled = true;

        const updateSounds = createSoundSystem();
        const updateTankMoveSounds = createTankMoveSoundSystem();

        // Load sounds asynchronously
        loadGameSounds().catch(console.error);

        SoundDI.soundFrame = (delta: number) => {
            updateSounds(delta);
            updateTankMoveSounds(delta);
        };

        SoundDI.destroy = () => {
            disposeSoundSystem();
            SoundManager.dispose();

            SoundDI.enabled = false;
            SoundDI.destroy = undefined;
            SoundDI.soundFrame = undefined;
        };
    }

    GameDI.setRenderTarget = async (canvas: null | undefined | HTMLCanvasElement) => {
        if (canvas === RenderDI.canvas) {
            return;
        }

        RenderDI.destroy?.();
        RenderDI.enabled = canvas != null;

        if (canvas == null) {
            return;
        }

        RenderDI.canvas = canvas;

        // Initialize camera position for first render
        setCameraPosition(CameraState.x, CameraState.y);

        const { device, context } = await initWebGPU(canvas);
        RenderDI.device = device;
        RenderDI.context = context;

        const drawGrass = createDrawGrassSystem();
        const drawMuzzleFlash = createDrawMuzzleFlashSystem();
        const drawHitFlash = createDrawHitFlashSystem();
        const drawExplosion = createDrawExplosionSystem();
        const drawExhaustSmoke = createDrawExhaustSmokeSystem();
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
            drawExhaustSmoke(passEncoder);
            drawExplosion(passEncoder);
            drawHitFlash(passEncoder);
            drawMuzzleFlash(passEncoder);
        });
        const postEffectFrame = createPostEffect(device, context, renderTexture);

        RenderDI.renderFrame = (delta: number) => {
            const commandEncoder = device.createCommandEncoder();
            renderFrame(commandEncoder, delta);
            postEffectFrame(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        };

        RenderDI.destroy = () => {
            RenderDI.enabled = false;
            RenderDI.canvas = null!;
            RenderDI.device = null!;
            RenderDI.context = null!;
            RenderDI.renderFrame = null!;
        };
    };

    GameDI.enablePlayer = () => {
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

            PlayerEnvDI.tankEid = null;
            PlayerEnvDI.playerId = null;
            PlayerEnvDI.document = null!;
            PlayerEnvDI.window = null!;
            PlayerEnvDI.destroy = null!;
            PlayerEnvDI.inputFrame = null!;
        };
    };
    GameDI.setPlayerId = (playerId: null | EntityId) => {
        PlayerEnvDI.playerId = playerId;
    };
    GameDI.setPlayerTank = (tankEid: null | EntityId) => {
        PlayerEnvDI.tankEid = tankEid;
    };
    GameDI.setCameraTarget = (tankEid: null | EntityId) => {
        setCameraTarget(tankEid);
    };
    GameDI.setInfiniteMapMode = (enabled: boolean) => {
        setInfiniteMapMode(enabled);
    };

    return GameDI;
}