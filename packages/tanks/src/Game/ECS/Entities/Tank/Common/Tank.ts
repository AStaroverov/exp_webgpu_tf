import { GameDI } from '../../../../DI/GameDI.ts';
import { Firearms } from '../../../Components/Firearms.ts';
import { Tank } from '../../../Components/Tank.ts';
import { TrackSide } from '../../../Components/Track.ts';
import { createVehicleBase, createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { createTrack, TrackOptions } from '../../Track/createTrack.ts';
import { type TankOptions } from './Options.ts';

export type TankTracksConfig = {
    anchorX: number;
    leftAnchorY: number;
    rightAnchorY: number;
    trackWidth: number;
    trackHeight: number;
};

export function createTankBase(options: TankOptions, { world } = GameDI): [number, number] {
    const [vehicleEid, vehiclePid] = createVehicleBase(options);

    // Add Tank-specific components
    Tank.addComponent(world, vehicleEid);

    return [vehicleEid, vehiclePid];
}

/**
 * Creates left and right track entities for a tank.
 * Tracks are attached to the tank body via fixed joints.
 */
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
        density: options.density * 0.1, // Tracks are lighter than main body
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
): [number, number] {
    const [turretEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        tankEid,
        tankPid,
    );

    // Add Tank-specific components
    Tank.setTurretEid(tankEid, turretEid);

    // Add Firearms component for shooting capability
    Firearms.addComponent(world, turretEid);
    Firearms.setData(turretEid, options.firearms.bulletStartPosition, options.firearms.bulletCaliber);
    Firearms.setReloadingDuration(turretEid, options.firearms.reloadingDuration);

    return [turretEid, turretPid];
}
