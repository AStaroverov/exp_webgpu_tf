import { createWorld, World } from 'bitecs';
import { Color, Roundness, Thinness } from '../../../../renderer/src/ECS/Components/Common.ts';
import { Rope } from '../../../../renderer/src/ECS/Components/Rope.ts';
import { Shape } from '../../../../renderer/src/ECS/Components/Shape.ts';
import { GlobalTransform, LocalTransform } from '../../../../renderer/src/ECS/Components/Transform.ts';
import { Building, BuildingPart } from './Components/Building.ts';
import { Bullet } from './Components/Bullet.ts';
import { Children } from './Components/Children.ts';
import { Damagable } from './Components/Damagable.ts';
import { Debris } from './Components/Debris.ts';
import { Destroy, DestroyBySpeed, DestroyByTimeout } from './Components/Destroy.ts';
import { ExhaustPipe } from './Components/ExhaustPipe.ts';
import { Firearms } from './Components/Firearms.ts';
import { HeuristicsData } from './Components/HeuristicsData.ts';
import { Hitable } from './Components/Hitable.ts';
import { Impulse, ImpulseAtPoint, TorqueImpulse } from './Components/Impulse.ts';
import { Joint } from './Components/Joint.ts';
import { JointMotor } from './Components/JointMotor.ts';
import { LastHitters } from './Components/LastHitters.ts';
import { Obstacle } from './Components/Obstacle.ts';
import { Parent } from './Components/Parent.ts';
import { RigidBodyRef, RigidBodyState } from './Components/Physical.ts';
import { PlayerRef } from './Components/PlayerRef.ts';
import { Progress } from './Components/Progress.ts';
import { Rock, RockPart } from './Components/Rock.ts';
import { Score } from './Components/Score.ts';
import { Slot } from './Components/Slot.ts';
import { Sound, DestroyOnSoundFinish, SoundParentRelative } from './Components/Sound.ts';
import { Spice } from './Components/Spice.ts';
import { SpiceCollector } from './Components/SpiceCollector.ts';
import { Tank } from './Components/Tank.ts';
import { TeamRef } from './Components/TeamRef.ts';
import { Track } from './Components/Track.ts';
import { TreadMark } from './Components/TreadMark.ts';
import { TurretController } from './Components/TurretController.ts';
import { VFX } from './Components/VFX.ts';
import { Vehicle } from './Components/Vehicle.ts';
import { VehicleController } from './Components/VehicleController.ts';
import { VehiclePart, VehiclePartCaterpillar } from './Components/VehiclePart.ts';
import { VehicleTurret } from './Components/VehicleTurret.ts';
import { Wheel, WheelDrive, WheelSteerable } from './Components/Wheel.ts';
import { Pilot } from '../../Plugins/Pilots/Components/Pilot.ts';
import { TankInputTensor } from '../../Plugins/Pilots/Components/TankState.ts';
import { VehicleInputTensor } from '../../Plugins/Pilots/Components/VehicleState.ts';

export const gameComponents = {
    Building,
    BuildingPart,
    Bullet,
    Children,
    Color,
    Damagable,
    Debris,
    Destroy,
    DestroyBySpeed,
    DestroyByTimeout,
    DestroyOnSoundFinish,
    ExhaustPipe,
    Firearms,
    GlobalTransform,
    HeuristicsData,
    Hitable,
    Impulse,
    ImpulseAtPoint,
    Joint,
    JointMotor,
    LastHitters,
    LocalTransform,
    Obstacle,
    Parent,
    Pilot,
    PlayerRef,
    Progress,
    RigidBodyRef,
    RigidBodyState,
    Rock,
    RockPart,
    Rope,
    Roundness,
    Score,
    Shape,
    Slot,
    Sound,
    SoundParentRelative,
    Spice,
    SpiceCollector,
    Tank,
    TankInputTensor,
    TeamRef,
    Thinness,
    TorqueImpulse,
    Track,
    TreadMark,
    TurretController,
    VFX,
    Vehicle,
    VehicleController,
    VehicleInputTensor,
    VehiclePart,
    VehiclePartCaterpillar,
    VehicleTurret,
    Wheel,
    WheelDrive,
    WheelSteerable,
};

export type GameWorld = World<{
    components: typeof gameComponents;
    time: {
        delta: number;
        elapsed: number;
        then: number;
    };
}>;

export function createGameWorld(): GameWorld {
    return createWorld({
        components: gameComponents,
        time: {
            delta: 0,
            elapsed: 0,
            then: performance.now(),
        },
    });
}
