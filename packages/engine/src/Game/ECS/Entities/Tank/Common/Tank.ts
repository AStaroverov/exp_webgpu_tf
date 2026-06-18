import { addEntity } from "bitecs";
import { GameDI } from "../../../../DI/GameDI.ts";
import {
  addTransformComponents,
  LocalTransform,
  setMatrixTranslate,
} from "renderer/src/ECS/Components/Transform.ts";
import { TrackSide } from "../../../Components/Track.ts";
import { createVehicleBase, createVehicleTurret } from "../../Vehicle/VehicleBase.ts";
import { createTrack, TrackOptions } from "../../Track/createTrack.ts";
import { type TankOptions } from "./Options.ts";
import { getGameComponents } from "../../../createGameWorld.ts";

export type TankTracksConfig = {
  anchorX: number;
  leftAnchorY: number;
  rightAnchorY: number;
  trackWidth: number;
  trackHeight: number;
};

export function createTankBase(options: TankOptions, { world } = GameDI): [number, number] {
  const { Tank, ActionsQueue } = getGameComponents(world);
  const [vehicleEid, vehiclePid] = createVehicleBase(options);
  Tank.addComponent(world, vehicleEid);
  // Decision #3 (not lazy): every tank gets its action queue so
  // query([ActionsQueue, Vehicle, RigidBodyState]) finds it.
  ActionsQueue.addComponent(world, vehicleEid);
  return [vehicleEid, vehiclePid];
}

export function createTankTracks(
  options: TankOptions,
  tracksConfig: TankTracksConfig,
  tankEid: number,
  tankPid: number,
): [leftTrackEid: number, rightTrackEid: number] {
  const trackOptions: TrackOptions = {
    ...options,
    width: tracksConfig.trackWidth,
    height: tracksConfig.trackHeight,
    trackSide: TrackSide.Left,
    trackLength: options.trackLength,
    anchorX: tracksConfig.anchorX,
    anchorY: tracksConfig.leftAnchorY,
    density: options.density * 0.1,
  };

  trackOptions.trackSide = TrackSide.Left;
  trackOptions.anchorY = tracksConfig.leftAnchorY;
  trackOptions.x = options.x;
  trackOptions.y = options.y;
  const [leftTrackEid] = createTrack(trackOptions, tankEid, tankPid);

  trackOptions.trackSide = TrackSide.Right;
  trackOptions.anchorY = tracksConfig.rightAnchorY;
  trackOptions.x = options.x;
  trackOptions.y = options.y;
  const [rightTrackEid] = createTrack(trackOptions, tankEid, tankPid);

  return [leftTrackEid, rightTrackEid];
}

export function createTankTurret(
  options: TankOptions,
  tankEid: number,
  tankPid: number,
  { world } = GameDI,
) {
  const { Firearms } = getGameComponents(world);

  const [turretEid, gunEid] = createTankTurretBase(options, tankEid, tankPid);

  Firearms.addComponent(world, turretEid, options.firearms.bulletCaliber);

  return [turretEid, gunEid] as const;
}

/**
 * Turret armed with a stream weapon (flame / frost hose): `StreamFirearms`
 * instead of `Firearms` — the spawners stay disjoint by component.
 */
export function createStreamTankTurret(
  options: TankOptions,
  tankEid: number,
  tankPid: number,
  caliber: number,
  { world } = GameDI,
) {
  const { StreamFirearms } = getGameComponents(world);

  const [turretEid, gunEid] = createTankTurretBase(options, tankEid, tankPid);

  StreamFirearms.addComponent(world, turretEid, caliber);

  return [turretEid, gunEid] as const;
}

function createTankTurretBase(
  options: TankOptions,
  tankEid: number,
  tankPid: number,
  { world } = GameDI,
) {
  const { Tank, SpawnDeltaPosition } = getGameComponents(world);

  const [turretEid, turretPid] = createVehicleTurret(options, options.turret, tankEid, tankPid);

  const gunEid = createTankGun(options, turretEid, turretPid);

  Tank.setTurretEid(tankEid, turretEid);
  SpawnDeltaPosition.addComponent(
    world,
    turretEid,
    options.spawnDeltaPosition[0],
    options.spawnDeltaPosition[1],
  );

  return [turretEid, gunEid] as const;
}

export function createTankGun(
  options: TankOptions,
  turretEid: number,
  _turretPid: number,
  { world } = GameDI,
): number {
  const { Parent, Children } = getGameComponents(world);

  const gunEid = addEntity(world);
  addTransformComponents(world, gunEid);
  setMatrixTranslate(
    LocalTransform.matrix.getBatch(gunEid),
    options.width / 2 + options.turret.gunWidth / 2,
    0,
  );

  Parent.addComponent(world, gunEid, turretEid);
  Children.addComponent(world, gunEid);
  Children.addChildren(turretEid, gunEid);

  return gunEid;
}
