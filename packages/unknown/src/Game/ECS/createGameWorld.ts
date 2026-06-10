import { createWorld, World } from "bitecs";
import { createRenderComponents, RenderComponents } from "../../../../renderer/src/ECS/world.ts";

import { createActionsQueueComponent } from "./Components/ActionsQueue.ts";
import { createBulletComponent } from "./Components/Bullet.ts";

import { createSlowedComponent } from "./Components/Slowed.ts";
import { createChildrenComponent } from "./Components/Children.ts";
import { createDamagableComponent } from "./Components/Damagable.ts";
import {
  createDestroyComponent,
  createDestroyByDistanceComponent,
  createDestroyByTimeoutComponent,
} from "./Components/Destroy.ts";
import { createExhaustPipeComponent } from "./Components/ExhaustPipe.ts";
import { createExplodableComponent } from "./Components/Explodable.ts";
import { createHullAimedComponent } from "./Components/HullAimed.ts";
import { createFirearmsComponent } from "./Components/Firearms.ts";
import { createSpawnDeltaPositionComponent } from "./Components/SpawnDeltaPosition.ts";
import { createHeuristicsDataComponent } from "./Components/HeuristicsData.ts";
import { createHitableComponent } from "./Components/Hitable.ts";
import {
  createImpulseComponent,
  createImpulseAtPointComponent,
  createTorqueImpulseComponent,
} from "./Components/Impulse.ts";
import { createJointComponent } from "./Components/Joint.ts";
import { createJointMotorComponent } from "./Components/JointMotor.ts";
import { createLastHittersComponent } from "./Components/LastHitters.ts";
import { createObstacleComponent } from "./Components/Obstacle.ts";
import { createObstacleFootprintComponent } from "./Components/ObstacleFootprint.ts";
import { createParentComponent } from "./Components/Parent.ts";
import {
  createRigidBodyRefComponent,
  createRigidBodyStateComponent,
} from "./Components/Physical.ts";
import { createPlayerRefComponent } from "./Components/PlayerRef.ts";
import { createProgressComponent } from "./Components/Progress.ts";
import { createSlotComponent } from "./Components/Slot.ts";
import {
  createSoundComponent,
  createDestroyOnSoundFinishComponent,
  createSoundParentRelativeComponent,
} from "./Components/Sound.ts";
import { createDotableComponent } from "./Components/Dotable.ts";
import { createDotComponent } from "./Components/Dot.ts";
import { createSensorHitsComponent } from "./Components/SensorHits.ts";
import { createOriginalColorComponent } from "./Components/OriginalColor.ts";
import { createLightEmitterAnimationComponent } from "./Components/LightEmitterAnimation.ts";
import { createStreamFirearmsComponent } from "./Components/StreamFirearms.ts";
import { createTankComponent } from "./Components/Tank.ts";
import { createTeamRefComponent } from "./Components/TeamRef.ts";
import { createTrackComponent } from "./Components/Track.ts";
import { createTreadMarkComponent } from "./Components/TreadMark.ts";
import { createTurretControllerComponent } from "./Components/TurretController.ts";
import { createVFXComponent } from "./Components/VFX.ts";
import { createVehicleComponent } from "./Components/Vehicle.ts";
import { createVehicleControllerComponent } from "./Components/VehicleController.ts";
import {
  createVehiclePartComponent,
  createVehiclePartCaterpillarComponent,
} from "./Components/VehiclePart.ts";
import { createVehicleTurretComponent } from "./Components/VehicleTurret.ts";
import { createWanderComponent } from "./Components/Wander.ts";
import {
  createWheelComponent,
  createWheelDriveComponent,
  createWheelSteerableComponent,
} from "./Components/Wheel.ts";

function createGameOnlyComponents(world: World) {
  return {
    ActionsQueue: createActionsQueueComponent(world),
    Bullet: createBulletComponent(world),
    Children: createChildrenComponent(world),
    Slowed: createSlowedComponent(world),
    Damagable: createDamagableComponent(world),
    Destroy: createDestroyComponent(world),
    DestroyByDistance: createDestroyByDistanceComponent(world),
    DestroyByTimeout: createDestroyByTimeoutComponent(world),
    DestroyOnSoundFinish: createDestroyOnSoundFinishComponent(world),
    ExhaustPipe: createExhaustPipeComponent(world),
    Explodable: createExplodableComponent(world),
    Firearms: createFirearmsComponent(world),
    SpawnDeltaPosition: createSpawnDeltaPositionComponent(world),
    HeuristicsData: createHeuristicsDataComponent(world),
    HullAimed: createHullAimedComponent(world),
    Hitable: createHitableComponent(world),
    Impulse: createImpulseComponent(world),
    ImpulseAtPoint: createImpulseAtPointComponent(world),
    Joint: createJointComponent(world),
    JointMotor: createJointMotorComponent(world),
    LastHitters: createLastHittersComponent(world),
    Obstacle: createObstacleComponent(world),
    ObstacleFootprint: createObstacleFootprintComponent(world),
    Parent: createParentComponent(world),
    PlayerRef: createPlayerRefComponent(world),
    Progress: createProgressComponent(world),
    RigidBodyRef: createRigidBodyRefComponent(world),
    RigidBodyState: createRigidBodyStateComponent(world),
    Slot: createSlotComponent(world),
    Sound: createSoundComponent(world),
    SoundParentRelative: createSoundParentRelativeComponent(world),
    Dotable: createDotableComponent(world),
    Dot: createDotComponent(world),
    SensorHits: createSensorHitsComponent(world),
    OriginalColor: createOriginalColorComponent(world),
    LightEmitterAnimation: createLightEmitterAnimationComponent(world),
    StreamFirearms: createStreamFirearmsComponent(world),
    Tank: createTankComponent(world),
    TeamRef: createTeamRefComponent(world),
    TorqueImpulse: createTorqueImpulseComponent(world),
    Track: createTrackComponent(world),
    TreadMark: createTreadMarkComponent(world),
    TurretController: createTurretControllerComponent(world),
    VFX: createVFXComponent(world),
    Vehicle: createVehicleComponent(world),
    VehicleController: createVehicleControllerComponent(world),
    VehiclePart: createVehiclePartComponent(world),
    VehiclePartCaterpillar: createVehiclePartCaterpillarComponent(world),
    VehicleTurret: createVehicleTurretComponent(world),
    Wander: createWanderComponent(world),
    Wheel: createWheelComponent(world),
    WheelDrive: createWheelDriveComponent(world),
    WheelSteerable: createWheelSteerableComponent(world),
  };
}

export type GameComponents = RenderComponents & ReturnType<typeof createGameOnlyComponents>;

export type GameWorld = World<{
  components: GameComponents;
  time: {
    delta: number;
    elapsed: number;
    then: number;
  };
}>;

export function createGameWorld(): GameWorld {
  const context = {
    components: null as unknown as GameComponents,
    time: {
      delta: 0,
      elapsed: 0,
      then: performance.now(),
    },
  };
  const world = createWorld(context) as GameWorld;
  context.components = {
    ...createRenderComponents(world),
    ...createGameOnlyComponents(world),
  };
  return world;
}

export function getGameComponents(world: World): GameComponents {
  const components = (world as GameWorld).components;
  if (!components) {
    throw new Error("Game components are not available on this world");
  }
  return components;
}
