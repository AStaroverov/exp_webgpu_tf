import { EventQueue } from '@dimforge/rapier2d-simd';
import { deleteWorld, EntityId, hasComponent, resetWorld } from 'bitecs';
import { destroyChangeDetectorSystem } from '../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDrawShapeSystem } from '../../../renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { createTransformSystem } from '../../../renderer/src/ECS/Systems/TransformSystem.ts';
import { initWebGPU } from '../../../renderer/src/gpu.ts';
import { createFrameTextures, createFrameTick } from '../../../renderer/src/WGSL/createFrame.ts';
import { GameDI } from './DI/GameDI.ts';
import { RenderDI } from './DI/RenderDI.ts';
import { getEntityIdByPhysicalId } from './ECS/Components/Physical.ts';
import { getGameComponents } from './ECS/createGameWorld.ts';
import { GameSession } from './ECS/Entities/GameSession.ts';
import { GameMap } from './ECS/Entities/GameMap.ts';
import { createTank } from './ECS/Entities/Tank/createTank.ts';
import { SystemGroup } from './ECS/Plugins/systems.ts';
import { createApplyRigidBodyToTransformSystem } from './ECS/Systems/createApplyRigidBodyToTransformSystem.ts';
import { createSpawnerBulletsSystem } from './ECS/Systems/createBulletSystem.ts';
import { createDestroyByTimeoutSystem } from './ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createDestroyBySpeedSystem } from './ECS/Systems/createDestroyBySpeedSystem.ts';
import { createDestroyOutOfZoneSystem } from './ECS/Systems/createDestroyOutOfZoneSystem.ts';
import { createDestroySystem } from './ECS/Systems/createDestroySystem.ts';
import { createRigidBodyStateSystem } from './ECS/Systems/createRigidBodyStateSystem.ts';
import { createApplyImpulseSystem } from './ECS/Systems/createApplyImpulseSystem.ts';
import { createDrawFaunaSystem } from './ECS/Systems/Render/Fauna/createDrawFaunaSystem.ts';
import { createSandstormSystem } from './ECS/Systems/Render/PostEffect/Sandstorm/createSandstormSystem.ts';
import { createDrawVFXSystem } from './ECS/Systems/Render/VFX/createDrawVFXSystem.ts';
import { createPostEffect } from './ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts';
import { createVisualizationTracksSystem } from './ECS/Systems/Tank/createVisualizationTracksSystem.ts';
import { createSpawnTreadMarksSystem } from './ECS/Systems/Tank/createSpawnTreadMarksSystem.ts';
import { createUpdateTreadMarksSystem } from './ECS/Systems/Tank/createUpdateTreadMarksSystem.ts';
import { createSpawnWheelTreadMarksSystem } from './ECS/Systems/Vehicle/createSpawnWheelTreadMarksSystem.ts';
import { createExhaustSmokeSpawnSystem } from './ECS/Systems/Vehicle/createExhaustSmokeSpawnSystem.ts';
import { createTrackControlSystem } from './ECS/Systems/Vehicle/TrackControlSystem.ts';
import { createWheelControlSystem } from './ECS/Systems/Vehicle/WheelControlSystem.ts';
import { createJointMotorSystem } from './ECS/Systems/Physical/createJointMotorSystem.ts';
import { initPhysicalWorld } from './Physical/initPhysicalWorld.ts';
import { createProgressSystem } from './ECS/Systems/createProgressSystem.ts';
import { createCameraSystem, setCameraTarget, initCameraPosition, CameraState } from './ECS/Systems/Camera/CameraSystem.ts';
import { setCameraPosition } from '../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { createSoundSystem, loadGameSounds, disposeSoundSystem, SoundManager, createTankMoveSoundSystem } from './ECS/Systems/Sound/index.ts';
import { createHitableSystem } from './ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './ECS/Systems/Tank/createTankAliveSystem.ts';
import { createShieldRegenerationSystem } from './ECS/Systems/createShieldRegenerationSystem.ts';
import { SoundDI } from './DI/SoundDI.ts';
import { createVehicleTurretRotationSystem as createTurretRotationSystem } from './ECS/Systems/Vehicle/VehicleControllerSystems.ts';
import { createGameWorld } from './ECS/createGameWorld.ts';
import { VehicleType } from './Config/index.ts';

export type Game = ReturnType<typeof createGame>;

export function createGame({ width, height }: {
    width: number,
    height: number,
}) {
    const world = createGameWorld();
    const physicalWorld = initPhysicalWorld();
    const { Children, Hitable, RigidBodyRef, Tank, TurretController, VehicleController } = getGameComponents(world);

    GameDI.width = width;
    GameDI.height = height;
    GameDI.world = world;
    GameDI.physicalWorld = physicalWorld;

    GameMap.setOffset(width / 2, height / 2);
    initCameraPosition();

    const execTransformSystem = createTransformSystem(world, Children);

    const updateTrackControl = createTrackControlSystem();
    const updateWheelControl = createWheelControlSystem();
    const updateTurretRotation = createTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyRigidBodyDeltaToLocalTransform = createApplyRigidBodyToTransformSystem();
    const applyImpulses = createApplyImpulseSystem();
    const applyJointMotors = createJointMotorSystem();

    const eventQueue = new EventQueue(true);

    const physicalFrame = (delta: number) => {
        updateTrackControl(delta);
        updateWheelControl(delta);
        updateTurretRotation(delta);

        execTransformSystem();
        applyImpulses();
        applyJointMotors(delta);
        physicalWorld.step(eventQueue);
        syncRigidBodyState();
        applyRigidBodyDeltaToLocalTransform();

        eventQueue.drainContactForceEvents(event => {
            const handle1 = event.collider1();
            const handle2 = event.collider2();
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
    const destroyFrame = (delta: number) => {
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
    const regenerateShields = createShieldRegenerationSystem();

    GameDI.gameTick = (delta: number) => {
        if (GameDI.world === null) return;

        physicalFrame(delta);

        GameDI.plugins.systems[SystemGroup.Before].forEach(system => system(delta));

        updateHitableSystem(delta);
        updateTankAliveSystem();
        regenerateShields(delta);

        visTracksUpdate(delta);
        updateProgress(delta);
        updateTreadMarks();

        updateCamera(delta);
        setCameraPosition(CameraState.x, CameraState.y);

        destroyFrame(delta);
        spawnFrame(delta);

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
    };

    GameDI.enableSound = async () => {
        if (SoundDI.enabled) {
            return;
        }

        SoundDI.enabled = true;

        const updateSounds = createSoundSystem();
        const updateTankMoveSounds = createTankMoveSoundSystem();

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
    };

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

        setCameraPosition(CameraState.x, CameraState.y);

        const { device, context } = await initWebGPU(canvas);
        RenderDI.device = device;
        RenderDI.context = context;

        const textures = createFrameTextures(device, canvas);

        const shapeSystem = createDrawShapeSystem({
            world,
            device,
            shadowMapTexture: textures.shadowMapTexture,
        });
        const drawFauna = createDrawFaunaSystem();
        const drawSandstorm = createSandstormSystem();
        const drawVFX = createDrawVFXSystem();

        const frameTick = createFrameTick(
            {
                ...textures,
                canvas,
                device,
                background: [226, 192, 146, 255].map(v => v / 255),
                getPixelRatio: () => window.devicePixelRatio,
            },
            ({ passEncoder, delta }) => {
                drawFauna(passEncoder, delta);
                shapeSystem.drawShapes(passEncoder);
                drawVFX(passEncoder);
                drawSandstorm(passEncoder, delta);
            },
            ({ passEncoder: shadowMapPassEncoder }) => {
                shapeSystem.drawShadowMap(shadowMapPassEncoder);
            },
        );

        const postEffectFrame = createPostEffect(device, context, textures.renderTexture);

        RenderDI.renderFrame = (delta: number) => {
            const commandEncoder = device.createCommandEncoder();
            frameTick(commandEncoder, delta);
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

    GameDI.setCameraTarget = (tankEid: null | EntityId) => {
        setCameraTarget(tankEid);
    };

    spawnDemoTanks();

    return GameDI;

function spawnDemoTanks() {
    const palette: Array<[number, number, number, number]> = [
        [1.0, 0.4, 0.4, 1],
        [0.4, 0.7, 1.0, 1],
        [0.6, 1.0, 0.5, 1],
        [1.0, 0.9, 0.4, 1],
    ];
    const cx = width / 2;
    const cy = height / 2;
    const radius = 300;
    const layout = [
        { x: cx + radius, y: cy,          rotation:  Math.PI * 0.5,  rot:  0.35 },
        { x: cx - radius, y: cy,          rotation: -Math.PI * 0.5,  rot:  0.35 },
        { x: cx,          y: cy + radius, rotation:  Math.PI,        rot: -0.35 },
        { x: cx,          y: cy - radius, rotation:  0,              rot: -0.35 },
    ];

    let firstTankEid: EntityId | null = null;

    for (let i = 0; i < layout.length; i++) {
        const slot = layout[i];
        const tankEid = createTank({
            type: VehicleType.MediumTank,
            playerId: i + 1,
            teamId: (i % 2) + 1,
            x: slot.x,
            y: slot.y,
            rotation: slot.rotation,
            color: new Float32Array(palette[i % palette.length]),
        });

        VehicleController.setMove$(tankEid, 1);
        VehicleController.setRotate$(tankEid, slot.rot);

        const turretEid = Tank.turretEId[tankEid];
        if (turretEid) {
            TurretController.setShooting$(turretEid, 1);
        }

        if (firstTankEid === null) {
            firstTankEid = tankEid;
        }
    }

    if (firstTankEid !== null) {
        setCameraTarget(firstTankEid);
    }
}
}
