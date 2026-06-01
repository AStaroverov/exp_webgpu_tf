import { cos, sin } from '../../../../../../../lib/math.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeByPhysics } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { TrackSide } from '../../Components/Track.ts';
import { createTrack, TrackOptions } from '../Track/createTrack.ts';
import { createVehicleBase, createVehicleTurret } from '../Vehicle/VehicleBase.ts';
import { type HarvesterOptions } from './Options.ts';

export type HarvesterTracksConfig = {
    anchorX: number;
    leftAnchorY: number;
    rightAnchorY: number;
    trackWidth: number;
    trackHeight: number;
};

export function createHarvesterBase(options: HarvesterOptions, { brainWorld } = Worlds): [number, number, number] {
    const { Tank } = getBrainWorldComponents(brainWorld);
    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = createVehicleBase(options);
    Tank.addComponent(brainWorld, getNodeByPhysics(vehiclePhysEid));
    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

/**
 * Creates left and right track entities for a harvester.
 * Tracks are attached to the harvester body via fixed joints.
 */
export function createHarvesterTracks(
    options: HarvesterOptions,
    tracksConfig: HarvesterTracksConfig,
    harvesterRenderEid: number,
    harvesterPid: number,
): [leftTrackRenderEid: number, rightTrackRenderEid: number] {
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
    const leftWorldX = tracksConfig.anchorX * cos(options.rotation) - tracksConfig.leftAnchorY * sin(options.rotation);
    const leftWorldY = tracksConfig.anchorX * sin(options.rotation) + tracksConfig.leftAnchorY * cos(options.rotation);
    trackOptions.x = options.x + leftWorldX;
    trackOptions.y = options.y + leftWorldY;

    const [, leftTrackRenderEid] = createTrack(trackOptions, harvesterRenderEid, harvesterPid);

    // Create right track
    trackOptions.trackSide = TrackSide.Right;
    trackOptions.anchorY = tracksConfig.rightAnchorY;

    const rightWorldX = tracksConfig.anchorX * cos(options.rotation) - tracksConfig.rightAnchorY * sin(options.rotation);
    const rightWorldY = tracksConfig.anchorX * sin(options.rotation) + tracksConfig.rightAnchorY * cos(options.rotation);
    trackOptions.x = options.x + rightWorldX;
    trackOptions.y = options.y + rightWorldY;

    const [, rightTrackRenderEid] = createTrack(trackOptions, harvesterRenderEid, harvesterPid);

    return [leftTrackRenderEid, rightTrackRenderEid];
}

// Returns [turretRenderEid, turretPid]
export function createHarvesterTurret(
    options: HarvesterOptions,
    _harvesterPhysEid: number,
    harvesterRenderEid: number,
    harvesterPid: number,
): [number, number] {
    // createVehicleTurret links the turret node as a Brain child of the hull node;
    // the turret is found from the hull node via getTurretPhysOfHull.
    const [, turretRenderEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        harvesterRenderEid,
        harvesterPid,
    );

    return [turretRenderEid, turretPid];
}

