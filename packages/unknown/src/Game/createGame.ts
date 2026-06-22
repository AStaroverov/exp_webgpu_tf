import { EventQueue } from "@dimforge/rapier2d-simd";
import type { EntityId } from "bitecs";
import { deleteWorld, hasComponent, resetWorld } from "bitecs";
import { destroyChangeDetectorSystem } from "../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts";
import { createDrawShapeSystem } from "../../../renderer/src/ECS/Systems/SDFSystem/createDrawShapeSystem.ts";
import { createTransformSystem } from "../../../renderer/src/ECS/Systems/TransformSystem.ts";
import { initWebGPU } from "../../../renderer/src/gpu.ts";
import { createFrameTextures, createFrameTick } from "../../../renderer/src/WGSL/createFrame.ts";
import { GameDI } from "./DI/GameDI.ts";
import { RenderDI } from "./DI/RenderDI.ts";
import { getEntityIdByPhysicalId } from "./ECS/Components/Physical.ts";
import { getEntityIdByColliderId } from "./ECS/Components/CompoundPart.ts";
import { getGameComponents } from "./ECS/createGameWorld.ts";
import { GameSession } from "./ECS/Entities/GameSession.ts";
import { GameMap } from "./ECS/Entities/GameMap.ts";
import { SystemGroup } from "./ECS/Plugins/systems.ts";
import { createApplyRigidBodyToTransformSystem } from "./ECS/Systems/createApplyRigidBodyToTransformSystem.ts";
import { createAttachedTransformSystem } from "./ECS/Systems/createAttachedTransformSystem.ts";
import { createCompoundPartTransformSystem } from "./ECS/Systems/createCompoundPartTransformSystem.ts";
import { createSpawnerBulletsSystem } from "./ECS/Systems/createBulletSystem.ts";
import { createStreamFirearmsSystem } from "./ECS/Systems/createStreamFirearmsSystem.ts";
import { createApplySensorHitsSystem } from "./ECS/Systems/createApplySensorHitsSystem.ts";
import { createDotSystem } from "./ECS/Systems/createDotSystem.ts";
import { createExpirySystem } from "./ECS/Systems/createExpirySystem.ts";
import { createSlowedExpirySystem } from "./ECS/Systems/createSlowedExpirySystem.ts";
import { createStunArcsSystem } from "./ECS/Systems/Render/createStunArcsSystem.ts";
import { createDestroyByTimeoutSystem } from "./ECS/Systems/createDestroyByTimeoutSystem.ts";
import { createDestroyByDistanceSystem } from "./ECS/Systems/createDestroyByDistanceSystem.ts";
import { createDestroyOutOfZoneSystem } from "./ECS/Systems/createDestroyOutOfZoneSystem.ts";
import { createDestroySystem } from "./ECS/Systems/createDestroySystem.ts";
import { createExplodeSystem } from "./ECS/Systems/createExplodeSystem.ts";
import { createRigidBodyStateSystem } from "./ECS/Systems/createRigidBodyStateSystem.ts";
import { createWanderSystem } from "./ECS/Systems/createWanderSystem.ts";
import { createApplyImpulseSystem } from "./ECS/Systems/createApplyImpulseSystem.ts";
import { createDrawFaunaSystem } from "./ECS/Systems/Render/Fauna/createDrawFaunaSystem.ts";
import { createSandstormSystem } from "./ECS/Systems/Render/PostEffect/Sandstorm/createSandstormSystem.ts";
import { createDrawVFXSystem } from "./ECS/Systems/Render/VFX/createDrawVFXSystem.ts";
import { createTintSystem } from "./ECS/Systems/Render/createTintSystem.ts";
import { createLightEmitterAnimationSystem } from "./ECS/Systems/Render/createLightEmitterAnimationSystem.ts";
import { createPresent } from "../../../renderer/src/WGSL/createPresent.ts";
import { createRadianceCascadesSystem } from "./ECS/Systems/Render/Lighting/createRadianceCascadesSystem.ts";
import { createVisualizationTracksSystem } from "./ECS/Systems/Tank/createVisualizationTracksSystem.ts";
import { createSpawnTreadMarksSystem } from "./ECS/Systems/Tank/createSpawnTreadMarksSystem.ts";
import { createLimitTreadMarksSystem } from "./ECS/Systems/Tank/createLimitTreadMarksSystem.ts";
import { createUpdateTreadMarksSystem } from "./ECS/Systems/Tank/createUpdateTreadMarksSystem.ts";
import { createSpawnWheelTreadMarksSystem } from "./ECS/Systems/Vehicle/createSpawnWheelTreadMarksSystem.ts";
import { createExhaustSmokeSpawnSystem } from "./ECS/Systems/Vehicle/createExhaustSmokeSpawnSystem.ts";
import { createTrackControlSystem } from "./ECS/Systems/Vehicle/TrackControlSystem.ts";
import { createWheelControlSystem } from "./ECS/Systems/Vehicle/WheelControlSystem.ts";
import { createJointMotorSystem } from "./ECS/Systems/Physical/createJointMotorSystem.ts";
import { initPhysicalWorld } from "./Physical/initPhysicalWorld.ts";
import { createProgressSystem } from "./ECS/Systems/createProgressSystem.ts";
import {
  createCameraSystem,
  setCameraTarget,
  initCameraPosition,
  CameraState,
} from "./ECS/Systems/Camera/CameraSystem.ts";
import {
  setCameraPosition,
  setCameraZoom,
} from "../../../renderer/src/ECS/Systems/ResizeSystem.ts";
import {
  createSoundSystem,
  loadGameSounds,
  disposeSoundSystem,
  SoundManager,
  createTankMoveSoundSystem,
} from "./ECS/Systems/Sound/index.ts";
import { createHitableSystem } from "./ECS/Systems/createHitableSystem.ts";
import { createTankAliveSystem } from "./ECS/Systems/Tank/createTankAliveSystem.ts";
import { createShieldRegenerationSystem } from "./ECS/Systems/createShieldRegenerationSystem.ts";
import { SoundDI } from "./DI/SoundDI.ts";
import { createVehicleTurretRotationSystem as createTurretRotationSystem } from "./ECS/Systems/Vehicle/VehicleControllerSystems.ts";
import { createGameWorld } from "./ECS/createGameWorld.ts";
import { MapDI } from "./DI/MapDI.ts";
import { HexGrid } from "./Map/HexGrid.ts";
import { createDrawGridSystem } from "./ECS/Systems/Render/Grid/createDrawGridSystem.ts";
import { createRunExecutors } from "./ECS/Actions/registry.ts";
import { createActionSchedulerSystem } from "./ECS/Actions/systems/ActionScheduler.ts";
import { createGridOccupancySystem } from "./ECS/Systems/Map/createGridOccupancySystem.ts";
import { CONTACT_FORCE_TARGET } from "./Config/index.ts";
import { min } from "../../../../lib/math.ts";
import { DamageKind } from "./ECS/Components/Damagable.ts";
import { createPixelatePass } from "./ECS/Systems/Render/PostEffect/Pixelate/createPostEffect.ts";

export type Game = ReturnType<typeof createGame>;

export function createGame({
  width,
  height,
  cols,
  rows,
}: {
  width: number;
  height: number;
  cols?: number;
  rows?: number;
}) {
  const world = createGameWorld();
  const physicalWorld = initPhysicalWorld();
  const { Children, Hitable, Damagable, RigidBodyRef, SensorHits, CompoundPart } =
    getGameComponents(world);

  GameDI.width = width;
  GameDI.height = height;
  GameDI.world = world;
  GameDI.physicalWorld = physicalWorld;

  GameMap.setOffset(width / 2, height / 2);
  initCameraPosition();

  MapDI.grid = new HexGrid({ center: { x: width / 2, y: height / 2 }, cols, rows });

  // Camera: sit at the field center and zoom out so the whole grid fits on
  // screen (with a small margin). No target — the camera stays put.
  {
    const bounds = MapDI.grid.worldBounds();
    const fieldW = bounds.maxX - bounds.minX;
    const fieldH = bounds.maxY - bounds.minY;
    const margin = 1.2; // padding around the field (larger -> more zoomed out)
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
  const updateAttachedTransforms = createAttachedTransformSystem();
  const updateCompoundPartTransforms = createCompoundPartTransformSystem();
  const applyImpulses = createApplyImpulseSystem();
  const applyJointMotors = createJointMotorSystem();
  const applyWander = createWanderSystem();

  const eventQueue = new EventQueue(true);

  const resolveEid = (colliderHandle: number): EntityId | null => {
    const part = getEntityIdByColliderId(colliderHandle);
    if (part !== undefined) return part;
    const rb = physicalWorld.getCollider(colliderHandle)?.parent();
    return rb ? getEntityIdByPhysicalId(rb.handle) : null;
  };

  const physicalFrame = (delta: number) => {
    updateTrackControl(delta);
    updateWheelControl(delta);
    updateTurretRotation(delta);

    execTransformSystem();
    applyImpulses();
    applyJointMotors(delta);
    applyWander(delta);
    physicalWorld.step(eventQueue);
    syncRigidBodyState();
    applyRigidBodyDeltaToLocalTransform();
    updateAttachedTransforms();
    updateCompoundPartTransforms();

    eventQueue.drainContactForceEvents((event) => {
      const eid1 = resolveEid(event.collider1());
      const eid2 = resolveEid(event.collider2());

      if (eid1 == null || eid2 == null) return;

      // hit$ carries FINAL damage: contact saturation coeff × the source's Damagable.
      const forceCoeff = min(1, event.totalForceMagnitude() / CONTACT_FORCE_TARGET);

      if (hasComponent(world, eid1, Hitable)) {
        Hitable.hit$(eid1, eid2, forceCoeff * Damagable.damage.get(eid2), DamageKind.Physical);
      }
      if (hasComponent(world, eid2, Hitable)) {
        Hitable.hit$(eid2, eid1, forceCoeff * Damagable.damage.get(eid1), DamageKind.Physical);
      }
    });

    eventQueue.drainCollisionEvents((handle1, handle2, started) => {
      if (!started) return; // only entry overlaps

      const eid1 = resolveEid(handle1);
      const eid2 = resolveEid(handle2);

      if (eid1 == null || eid2 == null) return;

      if (hasComponent(world, eid1, SensorHits)) {
        SensorHits.hit$(eid1, eid2);
      }
      if (hasComponent(world, eid2, SensorHits)) {
        SensorHits.hit$(eid2, eid1);
      }
    });
  };

  const spawnBullets = createSpawnerBulletsSystem();
  const streamEmit = createStreamFirearmsSystem();
  const spawnTreadMarks = createSpawnTreadMarksSystem();
  const spawnWheelTreadMarks = createSpawnWheelTreadMarksSystem();
  const spawnExhaustSmoke = createExhaustSmokeSpawnSystem();
  const spawnStunArcs = createStunArcsSystem();
  const spawnFrame = (delta: number) => {
    spawnBullets(delta);
    streamEmit(delta);
    spawnTreadMarks(delta);
    spawnWheelTreadMarks(delta);
    spawnExhaustSmoke(delta);
    spawnStunArcs(delta);
  };

  const destroy = createDestroySystem();
  const destroyOutOfZone = createDestroyOutOfZoneSystem();
  const destroyByTimeout = createDestroyByTimeoutSystem();
  const destroyByDistance = createDestroyByDistanceSystem();
  const limitTreadMarks = createLimitTreadMarksSystem();
  const explode = createExplodeSystem();
  const destroyFrame = (delta: number) => {
    destroyByTimeout(delta);
    destroyByDistance();
    destroyOutOfZone();
    limitTreadMarks(); // cap the tread-mark population before destroy() reaps the marked
    explode(); // Explodable + Destroy → detonate, before the entities are removed.
    destroy();
  };

  const visTracksUpdate = createVisualizationTracksSystem();
  const updateTreadMarks = createUpdateTreadMarksSystem();
  const updateProgress = createProgressSystem();
  const updateCamera = createCameraSystem();
  const updateHitableSystem = createHitableSystem();
  const updateTankAliveSystem = createTankAliveSystem();
  const regenerateShields = createShieldRegenerationSystem();
  const applySensorHits = createApplySensorHitsSystem();
  const dotTick = createDotSystem();
  const slowedExpiry = createSlowedExpirySystem();
  const expiry = createExpirySystem();
  const statusTint = createTintSystem();
  const animateLightEmitters = createLightEmitterAnimationSystem();

  const updateGridOccupancy = createGridOccupancySystem();

  const runActionExecutors = createRunExecutors();
  const actionScheduler = createActionSchedulerSystem();
  const updateActions = (delta: number) => {
    runActionExecutors(delta); // each kind system: run every owner's front (slot 0) of its kind
    actionScheduler(delta); // watchdog: time out stuck fronts; reaper: Finished front → slot 1 → slot 0, count--
  };

  // NOTE: the base game wires only systems. Build-specific world content — the
  // decision driver and the spawned entities (obstacles + units) — is added by
  // the caller: `setupDemoWorld()` for the dev game, `createUnknownScenario()`
  // (ppo_unknown) for training. This keeps createGame reusable across builds.

  GameDI.gameTick = (delta: number) => {
    if (GameDI.world === null) return;

    // Spawn at the top of the tick (consuming flags/requests raised last tick) so
    // freshly spawned entities flow through this tick's physics step and transform
    // resolution (execTransform → applyRigidBodyToTransform → attached) inside
    // `physicalFrame`. Spawning after that chain would leave their GlobalTransform at
    // identity until the next tick — rendering one frame at the origin (0,0).
    spawnFrame(delta);

    physicalFrame(delta);

    updateGridOccupancy(); // rebuild the grid's Unit/Reserved layer from vehicle state

    updateActions(delta);

    applySensorHits(); // drain the projectiles' overlap rings, hit + stamp Dots before hitable

    GameDI.plugins.systems[SystemGroup.Before].forEach((system) => system(delta));

    updateHitableSystem(delta);
    updateTankAliveSystem();
    regenerateShields(delta);
    dotTick(delta);
    slowedExpiry(delta);
    expiry(delta);

    visTracksUpdate(delta);
    statusTint(delta); // render-only: status recolor after the statuses settle
    updateProgress(delta);
    animateLightEmitters(); // render-only: decay light flashes from this frame's progress
    updateTreadMarks();

    updateCamera(delta);
    setCameraPosition(CameraState.x, CameraState.y);

    destroyFrame(delta);

    RenderDI.renderFrame?.(delta);
    SoundDI.soundFrame?.(delta);

    GameDI.plugins.systems[SystemGroup.After].forEach((system) => system(delta));
  };

  GameDI.destroy = () => {
    GameDI.plugins.dispose();

    physicalWorld.free();
    RigidBodyRef.dispose();
    CompoundPart.dispose();

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

    // Await the context resume (browser autoplay policy) before loading sounds,
    // so playback runs on a running context. enableSound is already async.
    await SoundManager.resume();
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
    const pixelatePass = createPixelatePass(device, textures.renderTexture);
    const lighting = (RenderDI.lighting = createRadianceCascadesSystem({
      device,
      frameTextures: textures,
      sceneTexture: pixelatePass.outputTexture,
      drawEmitters: shapeSystem.drawEmitters,
    }));
    const drawFauna = createDrawFaunaSystem();
    const drawGrid = createDrawGridSystem();
    const drawSandstorm = createSandstormSystem();
    const drawVFX = createDrawVFXSystem();

    const frameTick = createFrameTick(
      {
        ...textures,
        canvas,
        device,
        background: [226, 192, 146, 255].map((v) => v / 255),
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
        shapeSystem.prepare();
        shapeSystem.drawShadowMap(shadowMapPassEncoder);
      },
    );

    const present = createPresent(device, context);

    RenderDI.renderFrame = (delta: number) => {
      const commandEncoder = device.createCommandEncoder();
      frameTick(commandEncoder, delta); // scene -> renderTexture
      pixelatePass.run(commandEncoder); // renderTexture -> pixelated scene
      lighting.run(commandEncoder, delta); // pixelated scene * light -> litTexture
      present(commandEncoder, lighting.outputTexture); // final -> swapchain
      device.queue.submit([commandEncoder.finish()]);
    };

    RenderDI.destroy = () => {
      lighting.destroy();
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

  return GameDI;
}
