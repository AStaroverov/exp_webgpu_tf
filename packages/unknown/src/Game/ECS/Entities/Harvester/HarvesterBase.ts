import { cos, sin } from '../../../../../../../lib/math.ts';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
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

export function createHarvesterBase(world: PhysicsWorld, physicalWorld: PhysicalWorld, options: HarvesterOptions): [number, number, number] {
    const { Tank } = getPhysicsWorldComponents(world);
    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = createVehicleBase(world, physicalWorld, options);
    Tank.addComponent(world, vehiclePhysEid);
    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

/**
 * Creates left and right track entities for a harvester.
 * Tracks are attached to the harvester body via fixed joints.
 */
export function createHarvesterTracks(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
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

    const [, leftTrackRenderEid] = createTrack(world, physicalWorld, trackOptions, harvesterRenderEid, harvesterPid);

    // Create right track
    trackOptions.trackSide = TrackSide.Right;
    trackOptions.anchorY = tracksConfig.rightAnchorY;

    const rightWorldX = tracksConfig.anchorX * cos(options.rotation) - tracksConfig.rightAnchorY * sin(options.rotation);
    const rightWorldY = tracksConfig.anchorX * sin(options.rotation) + tracksConfig.rightAnchorY * cos(options.rotation);
    trackOptions.x = options.x + rightWorldX;
    trackOptions.y = options.y + rightWorldY;

    const [, rightTrackRenderEid] = createTrack(world, physicalWorld, trackOptions, harvesterRenderEid, harvesterPid);

    return [leftTrackRenderEid, rightTrackRenderEid];
}

// Returns [turretRenderEid, turretPid]
export function createHarvesterTurret(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: HarvesterOptions,
    harvesterPhysEid: number,
    harvesterRenderEid: number,
    harvesterPid: number,
): [number, number] {
    const { Tank } = getPhysicsWorldComponents(world);
    const [turretPhysEid, turretRenderEid, turretPid] = createVehicleTurret(
        world,
        physicalWorld,
        options,
        options.turret,
        harvesterRenderEid,
        harvesterPid,
    );

    Tank.setTurretEid(harvesterPhysEid, turretPhysEid);

    return [turretRenderEid, turretPid];
}

