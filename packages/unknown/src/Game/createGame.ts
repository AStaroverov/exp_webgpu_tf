import { EventQueue } from '@dimforge/rapier2d-simd';
import { deleteWorld, EntityId, hasComponent, resetWorld } from 'bitecs';
import { destroyChangeDetectorSystem } from '../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { createDrawShapeSystem } from '../../../renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts';
import { createTransformSystem } from '../../../renderer/src/ECS/Systems/TransformSystem.ts';
import { initWebGPU } from '../../../renderer/src/gpu.ts';
import { createFrameTextures, createFrameTick } from '../../../renderer/src/WGSL/createFrame.ts';
import { GameDI } from './DI/GameDI.ts';
import { Worlds } from './DI/Worlds.ts';
import { getNodeByPhysics } from './ECS/refs.ts';
import { physicsByBody } from './DI/physicsByBody.ts';
import { RenderDI } from './DI/RenderDI.ts';
import { getPhysicsWorldComponents, createPhysicsWorld } from './ECS/createPhysicsWorld.ts';
import { getRenderWorldComponents, createRenderWorld } from './ECS/createRenderWorld.ts';
import { createSlotWorld } from './ECS/createSlotWorld.ts';
import { createBrainWorld, getBrainWorldComponents } from './ECS/createBrainWorld.ts';
import { createFxWorld } from './ECS/createFxWorld.ts';
import { createSoundWorld } from './ECS/createSoundWorld.ts';
import { GameSession } from './ECS/Entities/GameSession.ts';
import { GameMap } from './ECS/Entities/GameMap.ts';
import { createTank } from './ECS/Entities/Tank/createTank.ts';
import { SystemGroup } from './ECS/Plugins/systems.ts';
import { createMirrorSyncSystem } from './ECS/Systems/createMirrorSyncSystem.ts';
import { createSpawnerBulletsSystem } from './ECS/Systems/createBulletSystem.ts';
import { createDestroyByTimeoutSystem } from './ECS/Systems/createDestroyByTimeoutSystem.ts';
import { createDestroyBySpeedSystem } from './ECS/Systems/createDestroyBySpeedSystem.ts';
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
import { VehicleType } from './Config/index.ts';
import { MapDI } from './DI/MapDI.ts';
import { HexGrid } from './Map/HexGrid.ts';
import { findPath } from './Map/findPath.ts';
import { createDrawGridSystem } from './ECS/Systems/Render/Grid/createDrawGridSystem.ts';
import { ActionScheduleDI } from './ECS/Actions/ActionScheduleDI.ts';
import { createActionWorld } from './ECS/Actions/createActionWorld.ts';
import { createRunExecutors } from './ECS/Actions/registry.ts';
import { createActionSchedulerSystem } from './ECS/Actions/createActionSchedulerSystem.ts';
import { enqueueAction } from './ECS/Actions/ActionSchedule.ts';
import { ActionKind, TargetKind } from './ECS/Actions/ActionTypes.ts';

export type Game = ReturnType<typeof createGame>;

export function createGame({ width, height }: {
    width: number,
    height: number,
}) {
    const physicsWorld = createPhysicsWorld();
    const renderWorld = createRenderWorld();
    const slotWorld = createSlotWorld();
    const brainWorld = createBrainWorld();
    const fxWorld = createFxWorld();
    const soundWorld = createSoundWorld();
    const physicalWorld = initPhysicalWorld();

    const { Hitable } = getPhysicsWorldComponents(physicsWorld);
    const { VehicleController } = getBrainWorldComponents(brainWorld);
    const { Children } = getRenderWorldComponents(renderWorld);

    GameDI.width = width;
    GameDI.height = height;
    Worlds.physicsWorld = physicsWorld;
    Worlds.renderWorld = renderWorld;
    Worlds.slotWorld = slotWorld;
    Worlds.brainWorld = brainWorld;
    Worlds.fxWorld = fxWorld;
    Worlds.soundWorld = soundWorld;
    Worlds.physicalWorld = physicalWorld;

    // Actions live in their own ECS world, separate from the game world.
    Worlds.actionWorld = createActionWorld();

    GameMap.setOffset(width / 2, height / 2);
    initCameraPosition();

    MapDI.grid = new HexGrid({ center: { x: width / 2, y: height / 2 } });

    const execTransformSystem = createTransformSystem(renderWorld, Children);
    const mirrorSync = createMirrorSyncSystem();

    const updateTrackControl = createTrackControlSystem();
    const updateWheelControl = createWheelControlSystem();
    const updateTurretRotation = createTurretRotationSystem();

    const syncRigidBodyState = createRigidBodyStateSystem();
    const applyImpulses = createApplyImpulseSystem();
    const applyJointMotors = createJointMotorSystem();

    const eventQueue = new EventQueue(true);

    const physicalFrame = (delta: number) => {
        updateTrackControl(delta);
        updateWheelControl(delta);
        updateTurretRotation(delta);

        applyImpulses();
        applyJointMotors(delta);
        physicalWorld.step(eventQueue);
        syncRigidBodyState();

        eventQueue.drainContactForceEvents(event => {
            const handle1 = event.collider1();
            const handle2 = event.collider2();
            const rb1 = physicalWorld.getCollider(handle1).parent();
            const rb2 = physicalWorld.getCollider(handle2).parent();
            const eid1 = rb1 ? physicsByBody.get(rb1.handle) : undefined;
            const eid2 = rb2 ? physicsByBody.get(rb2.handle) : undefined;

            if (eid1 == null || eid2 == null) return;

            const forceMagnitude = event.totalForceMagnitude();

            if (hasComponent(physicsWorld, eid1, Hitable)) {
                Hitable.hit$(eid1, eid2, forceMagnitude);
            }
            if (hasComponent(physicsWorld, eid2, Hitable)) {
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
    const destroyByTimeout = createDestroyByTimeoutSystem();
    const destroyBySpeed = createDestroyBySpeedSystem();
    const destroyFrame = (delta: number) => {
        destroyByTimeout(delta);
        destroyBySpeed();
        destroy();
    };

    const visTracksUpdate = createVisualizationTracksSystem();
    const updateTreadMarks = createUpdateTreadMarksSystem();
    const updateProgress = createProgressSystem();
    const updateCamera = createCameraSystem();
    const updateHitableSystem = createHitableSystem();
    const updateTankAliveSystem = createTankAliveSystem();
    const regenerateShields = createShieldRegenerationSystem();

    const runActionExecutors = createRunExecutors();
    const actionScheduler = createActionSchedulerSystem();
    const updateActions = (delta: number) => {
        runActionExecutors(delta); // each kind system: if top is mine, run it & mutate its status
        actionScheduler();         // pop Finished top, delete its entity → next top surfaces
    };

    GameDI.gameTick = (delta: number) => {
        if (Worlds.physicsWorld === null) return;

        physicalFrame(delta);

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
        physicsByBody.clear();

        resetWorld(physicsWorld);
        deleteWorld(physicsWorld);
        resetWorld(renderWorld);
        deleteWorld(renderWorld);
        resetWorld(slotWorld);
        deleteWorld(slotWorld);
        resetWorld(brainWorld);
        deleteWorld(brainWorld);
        resetWorld(fxWorld);
        deleteWorld(fxWorld);
        resetWorld(soundWorld);
        deleteWorld(soundWorld);
        destroyChangeDetectorSystem(physicsWorld); // JointMotor detector
        destroyChangeDetectorSystem(renderWorld);  // Shape/Color/Roundness detectors
        destroyChangeDetectorSystem(fxWorld);      // fx Shape/Color/Roundness detectors

        GameSession.reset();
        GameMap.reset();
        MapDI.grid = null!;
        ActionScheduleDI.nextSeq = 1;

        Worlds.physicsWorld = null!;
        Worlds.renderWorld = null!;
        Worlds.slotWorld = null!;
        Worlds.brainWorld = null!;
        Worlds.fxWorld = null!;
        Worlds.soundWorld = null!;
        Worlds.physicalWorld = null!;
        Worlds.actionWorld = null!;

        GameDI.width = null!;
        GameDI.height = null!;
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
            world: renderWorld,
            device,
            shadowMapTexture: textures.shadowMapTexture,
        });
        // Second SDF pass for FxWorld (tread marks are real SDF rectangles).
        const fxShapeSystem = createDrawShapeSystem({
            world: fxWorld,
            device,
            shadowMapTexture: textures.shadowMapTexture,
        });
        const drawFauna = createDrawFaunaSystem();
        const drawGrid = createDrawGridSystem();
        const drawSandstorm = createSandstormSystem();
        const drawVFX = createDrawVFXSystem(device, fxWorld);

        // fx have no Parent/Children → flat Local->Global compose. The empty
        // Children stub keeps the hierarchy pass a no-op (no fx entity has it).
        const execFxTransform = createTransformSystem(fxWorld, { entitiesCount: [], entitiesIds: { get: () => 0 } });

        const frameTick = createFrameTick(
            {
                ...textures,
                canvas,
                device,
                background: [226, 192, 146, 255].map(v => v / 255),
                getPixelRatio: () => window.devicePixelRatio,
            },
            ({ passEncoder, delta }) => {
                // RENDER tick: sync atom transforms to mirrors, then compose Local->Global.
                mirrorSync();
                execTransformSystem();      // RenderWorld (hierarchy)
                execFxTransform();          // FxWorld: flat Local->Global (no children)

                drawFauna(passEncoder, delta);
                drawGrid(passEncoder);
                shapeSystem.drawShapes(passEncoder);    // SDF pass #1 — RenderWorld mirrors
                fxShapeSystem.drawShapes(passEncoder);  // SDF pass #2 — FxWorld tread marks
                drawVFX(passEncoder);                   // VFX pass — FxWorld
                drawSandstorm(passEncoder, delta);
            },
            ({ passEncoder: shadowMapPassEncoder }) => {
                shapeSystem.drawShadowMap(shadowMapPassEncoder);
                fxShapeSystem.drawShadowMap(shadowMapPassEncoder);
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

    let firstTankEid: EntityId | null = null;
    const placed: Array<{ eid: EntityId; q: number; r: number }> = [];

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

        const tankBrain = getNodeByPhysics(tankEid);
        VehicleController.setMove$(tankBrain, 0);
        VehicleController.setRotate$(tankBrain, 0);

        // Mark the cell as occupied by this tank (game world entity).
        grid.occupy(q, r, tankEid);
        placed.push({ eid: tankEid, q, r });

        if (firstTankEid === null) {
            firstTankEid = tankEid;
        }
    }

    if (firstTankEid !== null) {
        setCameraTarget(firstTankEid);
    }

    // Demo: enqueue MoveToHex actions onto the single global FIFO stack. Only the
    // top action runs at a time, so tanks move one after another (chess-like).
    // For each tank, chain a couple of hops to random reachable empty cells.
    for (let t = 0; t < placed.length; t++) {
        const tank = placed[t];
        let fromQ = tank.q;
        let fromR = tank.r;

        for (let hop = 0; hop < 2; hop++) {
            const target = pickReachableCell(fromQ, fromR);
            if (!target) break;

            enqueueAction(tank.eid, {
                kind: ActionKind.MoveToHex,
                target: { kind: TargetKind.Hex, q: target.q, r: target.r },
                params: { speed: 1 },
            });

            fromQ = target.q;
            fromR = target.r;

            // After moving, aim the turret at the next tank (circular) and fire a
            // couple of rounds — demonstrates the TurretAim + Fire actions on the
            // same global FIFO stack.
            const targetTank = placed[(t + 1) % placed.length];
            if (targetTank.eid !== tank.eid) {
                enqueueAction(tank.eid, {
                    kind: ActionKind.TurretAim,
                    target: { kind: TargetKind.Entity, eid: targetTank.eid },
                    params: { tolerance: 0.05 },
                });
                enqueueAction(tank.eid, {
                    kind: ActionKind.Fire,
                    params: { shots: 2 },
                });
            }
        }
    }

    // Pick a random walkable + empty cell reachable from (q, r) via A*.
    function pickReachableCell(fromQ: number, fromR: number): { q: number; r: number } | null {
        const candidates: Array<{ q: number; r: number }> = [];
        grid.forEachCell((cell) => {
            if (cell.q === fromQ && cell.r === fromR) return;
            if (!grid.isPassable(cell.q, cell.r)) return;
            candidates.push({ q: cell.q, r: cell.r });
        });
        for (let i = candidates.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
        }
        for (const c of candidates) {
            const path = findPath(grid, { q: fromQ, r: fromR }, { q: c.q, r: c.r });
            if (path && path.length > 1) return c;
        }
        return null;
    }
}
}
