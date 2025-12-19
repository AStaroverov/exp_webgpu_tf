import { cos, sin } from '../../../../../../../lib/math.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Tank } from '../../Components/Tank.ts';
import { TrackSide } from '../../Components/Track.ts';
import { createTrack, TrackOptions } from '../Track/createTrack.ts';
import { createVehicleBase, createVehicleTurret } from '../Vehicle/VehicleBase.ts';
import { type HarvesterOptions } from './Options.ts';

export type HarvesterTracksConfig = {
    leftAnchorX: number;
    rightAnchorX: number;
    anchorY: number;
    trackWidth: number;
    trackHeight: number;
};

export function createHarvesterBase(options: HarvesterOptions, { world } = GameDI): [number, number] {
    const [vehicleEid, vehiclePid] = createVehicleBase(options);

    // Add Tank component for caterpillars and turret tracking
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
        anchorX: tracksConfig.leftAnchorX,
        anchorY: tracksConfig.anchorY,
        density: options.density * 0.1, // Tracks are lighter than main body
    };

    // Transform anchor positions to world space
    const leftWorldX = tracksConfig.leftAnchorX * cos(options.rotation) - tracksConfig.anchorY * sin(options.rotation);
    const leftWorldY = tracksConfig.leftAnchorX * sin(options.rotation) + tracksConfig.anchorY * cos(options.rotation);
    trackOptions.x = options.x + leftWorldX;
    trackOptions.y = options.y + leftWorldY;

    const [leftTrackEid] = createTrack(trackOptions, harvesterEid, harvesterPid);

    // Create right track
    trackOptions.trackSide = TrackSide.Right;
    trackOptions.anchorX = tracksConfig.rightAnchorX;
    
    const rightWorldX = tracksConfig.rightAnchorX * cos(options.rotation) - tracksConfig.anchorY * sin(options.rotation);
    const rightWorldY = tracksConfig.rightAnchorX * sin(options.rotation) + tracksConfig.anchorY * cos(options.rotation);
    trackOptions.x = options.x + rightWorldX;
    trackOptions.y = options.y + rightWorldY;

    const [rightTrackEid] = createTrack(trackOptions, harvesterEid, harvesterPid);

    return [leftTrackEid, rightTrackEid];
}

export function createHarvesterTurret(
    options: HarvesterOptions,
    harvesterEid: number,
    harvesterPid: number,
): [number, number] {
    const [turretEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        harvesterEid,
        harvesterPid,
    );

    // Add Tank turret reference (no Firearms - harvester doesn't shoot)
    Tank.setTurretEid(harvesterEid, turretEid);

    return [turretEid, turretPid];
}

