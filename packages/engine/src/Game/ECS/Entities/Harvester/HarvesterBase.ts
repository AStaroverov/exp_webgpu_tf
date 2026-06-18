import { cos, sin } from "../../../../../../../lib/math.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { TrackSide } from "../../Components/Track.ts";
import { createTrack, TrackOptions } from "../Track/createTrack.ts";
import { createVehicleBase, createVehicleTurret } from "../Vehicle/VehicleBase.ts";
import { type HarvesterOptions } from "./Options.ts";

export type HarvesterTracksConfig = {
  anchorX: number;
  leftAnchorY: number;
  rightAnchorY: number;
  trackWidth: number;
  trackHeight: number;
};

export function createHarvesterBase(
  options: HarvesterOptions,
  { world } = GameDI,
): [number, number] {
  const { Tank } = getGameComponents(world);
  const [vehicleEid, vehiclePid] = createVehicleBase(options);
  Tank.addComponent(world, vehicleEid);
  return [vehicleEid, vehiclePid];
}

/**
 * Creates left and right track entities for a harvester.
 * Tracks are attached to the harvester body via fixed joints.
 */
export function createHarvesterTracks(
  options: HarvesterOptions,
  tracksConfig: HarvesterTracksConfig,
  harvesterEid: number,
  harvesterPid: number,
): [leftTrackEid: number, rightTrackEid: number] {
  const trackOptions: TrackOptions = {
    ...options,
    width: tracksConfig.trackWidth,
    height: tracksConfig.trackHeight,
    trackLength: options.trackLength,
    trackSide: TrackSide.Left,
    anchorX: tracksConfig.anchorX,
    anchorY: tracksConfig.leftAnchorY,
    density: options.density * 0.1, // Tracks are lighter than main body
  };

  // Transform anchor positions to world space
  const leftWorldX =
    tracksConfig.anchorX * cos(options.rotation) - tracksConfig.leftAnchorY * sin(options.rotation);
  const leftWorldY =
    tracksConfig.anchorX * sin(options.rotation) + tracksConfig.leftAnchorY * cos(options.rotation);
  trackOptions.x = options.x + leftWorldX;
  trackOptions.y = options.y + leftWorldY;

  const [leftTrackEid] = createTrack(trackOptions, harvesterEid, harvesterPid);

  // Create right track
  trackOptions.trackSide = TrackSide.Right;
  trackOptions.anchorY = tracksConfig.rightAnchorY;

  const rightWorldX =
    tracksConfig.anchorX * cos(options.rotation) -
    tracksConfig.rightAnchorY * sin(options.rotation);
  const rightWorldY =
    tracksConfig.anchorX * sin(options.rotation) +
    tracksConfig.rightAnchorY * cos(options.rotation);
  trackOptions.x = options.x + rightWorldX;
  trackOptions.y = options.y + rightWorldY;

  const [rightTrackEid] = createTrack(trackOptions, harvesterEid, harvesterPid);

  return [leftTrackEid, rightTrackEid];
}

export function createHarvesterTurret(
  options: HarvesterOptions,
  harvesterEid: number,
  harvesterPid: number,
  { world } = GameDI,
): [number, number] {
  const { Tank } = getGameComponents(world);
  const [turretEid, turretPid] = createVehicleTurret(
    options,
    options.turret,
    harvesterEid,
    harvesterPid,
  );

  Tank.setTurretEid(harvesterEid, turretEid);

  return [turretEid, turretPid];
}
