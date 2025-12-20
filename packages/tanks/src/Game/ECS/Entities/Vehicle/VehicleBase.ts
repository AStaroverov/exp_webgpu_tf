import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { Color } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { VehicleInputTensor } from '../../../../Pilots/Components/VehicleState.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { Children } from '../../Components/Children.ts';
import { HeuristicsData } from '../../Components/HeuristicsData.ts';
import { Parent } from '../../Components/Parent.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { createRectangleRigidGroup } from '../../Components/RigidGroup.ts';
import { ImpulseAtPoint } from '../../Components/Impulse.ts';
import { TurretController } from '../../Components/TurretController.ts';
import { Vehicle, VehicleType } from '../../Components/Vehicle.ts';
import { Joint } from '../../Components/Joint.ts';
import { JointMotor } from '../../Components/JointMotor.ts';
import { VehicleTurret } from '../../Components/VehicleTurret.ts';
import { TeamRef } from '../../Components/TeamRef.ts';
import { VehicleOptions } from './Options.ts';
import { spawnSoundAtParent } from '../Sound.ts';
import { SoundParentRelative, SoundType } from '../../Components/Sound.ts';
import { VehicleController } from '../../Components/VehicleController.ts';

const volumeByType: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 0.6,
    [VehicleType.MediumTank]: 0.8,
    [VehicleType.HeavyTank]: 1.0,
    [VehicleType.PlayerTank]: 0.8,
    [VehicleType.Harvester]: 1.0,
    [VehicleType.MeleeCar]: 0.7,
};

/**
 * Creates base vehicle entity with all common components.
 * Tracks and wheels are added as children.
 */
export function createVehicleBase(options: VehicleOptions, { world } = GameDI): [number, number] {
    options.belongsCollisionGroup = CollisionGroup.VEHICALE_BASE;
    options.interactsCollisionGroup = CollisionGroup.VEHICALE_BASE;

    const [vehicleEid, vehiclePid] = createRectangleRigidGroup(options);
    Vehicle.addComponent(world, vehicleEid, options.vehicleType);
    Vehicle.setEngineType(vehicleEid, options.engineType);

    addTransformComponents(world, vehicleEid);
    Children.addComponent(world, vehicleEid);
    TeamRef.addComponent(world, vehicleEid, options.teamId);
    PlayerRef.addComponent(world, vehicleEid, options.playerId);
    HeuristicsData.addComponent(world, vehicleEid, options.approximateColliderRadius);
    Color.addComponent(world, vehicleEid, ...options.color);

    VehicleInputTensor.addComponent(world, vehicleEid);

    // Add ImpulseAtPoint for realistic physics (force applied at track/wheel positions)
    ImpulseAtPoint.addComponent(world, vehicleEid);
    
    // All vehicles use VehicleController for movement input
    VehicleController.addComponent(world, vehicleEid);

    const soundEid = spawnSoundAtParent({
        parentEid: vehicleEid,
        type: SoundType.TankMove,
        volume: volumeByType[options.vehicleType] ?? 0.8,
        loop: true,
        autoplay: false,
    });
    SoundParentRelative.addComponent(world, soundEid);

    return [vehicleEid, vehiclePid];
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

export type TurretOptions = {
    rotationSpeed: number;
};

export function createVehicleTurret(
    options: VehicleOptions,
    turretOptions: TurretOptions,
    vehicleEid: number,
    vehiclePid: number,
    { world, physicalWorld } = GameDI,
): [number, number] {
    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;

    const [turretEid, turretPid] = createRectangleRigidGroup(options);
    VehicleTurret.addComponent(world, turretEid);
    VehicleTurret.setRotationSpeed(turretEid, turretOptions.rotationSpeed);
    TurretController.addComponent(world, turretEid);

    parentVector.x = 0;
    parentVector.y = 0;
    childVector.x = 0;
    childVector.y = 0;

    const joint = physicalWorld.createImpulseJoint(
        JointData.revolute(parentVector, childVector),
        physicalWorld.getRigidBody(vehiclePid),
        physicalWorld.getRigidBody(turretPid),
        false,
    );
    Joint.addComponent(world, turretEid, joint.handle);
    JointMotor.addComponent(world, turretEid);

    addTransformComponents(world, turretEid);
    Parent.addComponent(world, turretEid, vehicleEid);
    Children.addComponent(world, turretEid);
    Children.addChildren(vehicleEid, turretEid);

    return [turretEid, turretPid];
}

