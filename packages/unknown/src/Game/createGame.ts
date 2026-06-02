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
import { setCameraPosition, setCameraZoom } from '../../../renderer/src/ECS/Systems/ResizeSystem.ts';
import { createSoundSystem, loadGameSounds, disposeSoundSystem, SoundManager, createTankMoveSoundSystem } from './ECS/Systems/Sound/index.ts';
import { createHitableSystem } from './ECS/Systems/createHitableSystem.ts';
import { createTankAliveSystem } from './ECS/Systems/Tank/createTankAliveSystem.ts';
import { createShieldRegenerationSystem } from './ECS/Systems/createShieldRegenerationSystem.ts';
import { SoundDI } from './DI/SoundDI.ts';
import { createVehicleTurretRotationSystem as createTurretRotationSystem } from './ECS/Systems/Vehicle/VehicleControllerSystems.ts';
import { createGameWorld } from './ECS/createGameWorld.ts';
import { VehicleType } from './Config/index.ts';
import { MapDI } from './DI/MapDI.ts';
import { HexGrid } from './Map/HexGrid.ts';
import { spawnObstacles } from './ECS/Entities/Obstacle/spawnObstacles.ts';
import { createDrawGridSystem } from './ECS/Systems/Render/Grid/createDrawGridSystem.ts';
import { createRunExecutors } from './ECS/Actions/registry.ts';
import { createActionSchedulerSystem } from './ECS/Actions/systems/ActionScheduler.ts';
import { createStandInDriverSystem } from './ECS/Plugins/createStandInDriverSystem.ts';
import { createShapeCountDiagnosticSystem } from './ECS/Plugins/createShapeCountDiagnosticSystem.ts';
import { createGridOccupancySystem } from './ECS/Systems/Map/createGridOccupancySystem.ts';
import { PluginDI } from './DI/PluginDI.ts';

export type Game = ReturnType<typeof createGame>;

export function createGame({ width, height }: {
    width: number,
    height: number,
}) {
    const world = createGameWorld();
    const physicalWorld = initPhysicalWorld();
    const { Children, Hitable, RigidBodyRef, VehicleController } = getGameComponents(world);

    GameDI.width = width;
    GameDI.height = height;
    GameDI.world = world;
    GameDI.physicalWorld = physicalWorld;

    GameMap.setOffset(width / 2, height / 2);
    initCameraPosition();

    MapDI.grid = new HexGrid({ center: { x: width / 2, y: height / 2 } });

    // Camera: sit at the field center and zoom out so the whole grid fits on
    // screen (with a small margin). No target — the camera stays put.
    {
        const bounds = MapDI.grid.worldBounds();
        const fieldW = bounds.maxX - bounds.minX;
        const fieldH = bounds.maxY - bounds.minY;
        const margin = 2; // padding around the field (larger -> more zoomed out)
        const zoom = Math.min(width / (fieldW * margin), height / (fieldH * margin));
        setCameraPosition((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2);
        setCameraZoom(zoom);
    }

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

    const updateGridOccupancy = createGridOccupancySystem();

    const runActionExecutors = createRunExecutors();
    const actionScheduler = createActionSchedulerSystem();
    const updateActions = (delta: number) => {
        runActionExecutors(delta); // each kind system: run every owner's front (slot 0) of its kind
        actionScheduler();         // reaper: Finished front → shift slot 1 → slot 0, count--
    };

    // Stand-in decision driver — placeholder for the future ML policy. Runs in
    // SystemGroup.Before so decisions land before the gameplay/spawn systems act
    // on them this tick (PLAN.md §8). Same seam the ML driver will use.
    const standInDriver = createStandInDriverSystem();
    PluginDI.addSystem(SystemGroup.Before, standInDriver);

    // TEMP diagnostic (DELETE once the 10k shape-buffer overflow is diagnosed).
    PluginDI.addSystem(SystemGroup.After, createShapeCountDiagnosticSystem());

    GameDI.gameTick = (delta: number) => {
        if (GameDI.world === null) return;

        physicalFrame(delta);

        updateGridOccupancy(); // rebuild the grid's Unit/Reserved layer from vehicle state

        updateActions(delta);

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
        MapDI.grid = null!;

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
        const drawGrid = createDrawGridSystem();
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
                drawGrid(passEncoder);
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

    spawnObstacles();
    spawnDemoTanks();

    return GameDI;

    function spawnDemoTanks() {
        const grid = MapDI.grid;
        const palette: Array<[number, number, number, number]> = [
            [1.0, 0.4, 0.4, 1],
            [0.4, 0.7, 1.0, 1],
            [0.6, 1.0, 0.5, 1],
            [1.0, 0.9, 0.4, 1],
        ];

        // Pick distinct random cells to place the tanks on.
        const allCells: Array<{ q: number; r: number }> = [];
        grid.forEachCell((cell) => allCells.push({ q: cell.q, r: cell.r }));
        for (let i = allCells.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [allCells[i], allCells[j]] = [allCells[j], allCells[i]];
        }
        const slots = allCells.slice(0, palette.length);

        for (let i = 0; i < slots.length; i++) {
            const { q, r } = slots[i];
            const pos = grid.hexToWorld({ q, r });
            if (!pos) continue;

            const tankEid = createTank({
                type: VehicleType.MediumTank,
                playerId: i + 1,
                teamId: (i % 2) + 1,
                x: pos.x,
                y: pos.y,
                rotation: Math.random() * Math.PI * 2,
                color: new Float32Array(palette[i % palette.length]),
            });

            VehicleController.setMove$(tankEid, 0);
            VehicleController.setRotate$(tankEid, 0);
        }
    }
}
