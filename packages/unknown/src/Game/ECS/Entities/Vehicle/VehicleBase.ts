import { JointData, Vector2 } from "@dimforge/rapier2d-simd";
import { getGameComponents } from "../../createGameWorld.ts";
import { addTransformComponents } from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { CollisionGroup } from "../../../Physical/createRigid.ts";
import { createRectangleRigidGroup } from "../../Components/RigidGroup.ts";
import { VehicleType } from "../../Components/Vehicle.ts";
import { VehicleOptions } from "./Options.ts";
import { spawnSoundAtParent } from "../Sound.ts";
import { SoundType } from "../../Components/Sound.ts";

const volumeByType: Record<VehicleType, number> = {
  [VehicleType.LightTank]: 0.6,
  [VehicleType.MediumTank]: 0.8,
  [VehicleType.HeavyTank]: 1.0,
  [VehicleType.RocketTank]: 1.0,
  [VehicleType.Harvester]: 1.0,
  [VehicleType.MeleeCar]: 0.7,
  [VehicleType.FlameTank]: 0.8,
  [VehicleType.FrostTank]: 0.8,
  [VehicleType.EmpTank]: 0.8,
};

export function createVehicleBase(options: VehicleOptions, { world } = GameDI): [number, number] {
  const {
    Vehicle,
    Color,
    Children,
    TeamRef,
    PlayerRef,
    LastHitters,
    FriendlyHitters,
    ImpulseAtPoint,
    VehicleController,
    SoundParentRelative,
  } = getGameComponents(world);

  options.belongsCollisionGroup = CollisionGroup.VEHICALE_BASE;
  options.interactsCollisionGroup = CollisionGroup.VEHICALE_BASE;

  const [vehicleEid, vehiclePid] = createRectangleRigidGroup(options);
  Vehicle.addComponent(world, vehicleEid, options.vehicleType);
  Vehicle.setEngineType(vehicleEid, options.engineType);

  addTransformComponents(world, vehicleEid);
  Children.addComponent(world, vehicleEid);
  Color.addComponent(world, vehicleEid, ...options.color);
  TeamRef.addComponent(world, vehicleEid, options.teamId);
  PlayerRef.addComponent(world, vehicleEid, options.playerId);
  LastHitters.addComponent(world, vehicleEid);
  FriendlyHitters.addComponent(world, vehicleEid);

  ImpulseAtPoint.addComponent(world, vehicleEid);

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
  const { VehicleTurret, TurretController, Joint, JointMotor, Parent, Children } =
    getGameComponents(world);

  options.belongsCollisionGroup = 0;
  options.interactsCollisionGroup = 0;

  const [turretEid, turretPid] = createRectangleRigidGroup(options);
  VehicleTurret.addComponent(world, turretEid, turretOptions.rotationSpeed);
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
