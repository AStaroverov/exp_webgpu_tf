import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { TrackSide } from '../../../Components/Track.ts';
import { createVehicleBase, createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { createTrack, TrackOptions } from '../../Track/createTrack.ts';
import { type TankOptions } from './Options.ts';
import { spawnRectangleCarrier } from '../../spawnPart.ts';
import { getPhysicsWorldComponents } from '../../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../../createBrainWorld.ts';
import { getNodeByPhysics, setNodeRender, linkBrainChild } from '../../../refs.ts';
import { addEntity } from 'bitecs';
import { Worlds } from '../../../../DI/Worlds.ts';

export type TankTracksConfig = {
    anchorX: number;
    leftAnchorY: number;
    rightAnchorY: number;
    trackWidth: number;
    trackHeight: number;
};

// Returns [vehiclePhysEid, vehicleRenderEid, vehiclePid]
export function createTankBase(
    options: TankOptions,
    { brainWorld } = Worlds,
): [number, number, number] {
    const { Tank } = getBrainWorldComponents(brainWorld);
    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = createVehicleBase(options);
    // Tank lives on the hull-brain (the hull node whose presentation is the hull atom).
    Tank.addComponent(brainWorld, getNodeByPhysics(vehiclePhysEid));
    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

// Returns [leftTrackRenderEid, rightTrackRenderEid]
export function createTankTracks(
    options: TankOptions,
    tracksConfig: TankTracksConfig,
    tankRenderEid: number,
    tankPid: number,
): [leftTrackRenderEid: number, rightTrackRenderEid: number] {
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
    const [, leftTrackRenderEid] = createTrack(trackOptions, tankRenderEid, tankPid);

    trackOptions.trackSide = TrackSide.Right;
    trackOptions.anchorY = tracksConfig.rightAnchorY;
    trackOptions.x = options.x;
    trackOptions.y = options.y;
    const [, rightTrackRenderEid] = createTrack(trackOptions, tankRenderEid, tankPid);

    return [leftTrackRenderEid, rightTrackRenderEid];
}

export function createTankTurret(
    tankRenderEid: number,
    tankPid: number,
    options: TankOptions,
    { brainWorld } = Worlds,
): readonly [turretRenderEid: number, gunRenderEid: number] {
    const { Firearms } = getBrainWorldComponents(brainWorld);

    const [turretPhysEid, turretRenderEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        tankRenderEid,
        tankPid,
    );

    const [, gunRenderEid] = createTankGun(turretPid, turretPhysEid, options);

    const turretBrain = getNodeByPhysics(turretPhysEid);
    Firearms.addComponent(brainWorld, turretBrain);
    Firearms.setData(turretBrain, options.firearms.bulletStartPosition, options.firearms.bulletCaliber);
    Firearms.setReloadingDuration(turretBrain, options.firearms.reloadingDuration);

    return [turretRenderEid, gunRenderEid] as const;
}

export function createTankGun(
    turretPid: number,
    turretPhysEid: number,
    options: TankOptions,
    { physicsWorld, physicalWorld, brainWorld } = Worlds,
): [gunPhysEid: number, gunRenderEid: number, gunPid: number] {
    const { Joint } = getPhysicsWorldComponents(physicsWorld);

    // Brain-first: the gun node before its physics/render presentation.
    const gunNode = addEntity(brainWorld);

    const [gunPhysEid, gunRenderEid, gunPid] = spawnRectangleCarrier({
        ...options,
        width: options.turret.gunWidth,
        height: options.turret.gunHeight,
        belongsCollisionGroup: 0,
        interactsCollisionGroup: 0,
    });

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(
            new Vector2(options.width / 2, 0), 0,
            new Vector2(-options.turret.gunWidth / 2, 0), 0,
        ),
        physicalWorld.getRigidBody(turretPid),
        physicalWorld.getRigidBody(gunPid),
        false,
    );
    Joint.addComponent(physicsWorld, gunPhysEid, joint.handle);

    const turretBrain = getNodeByPhysics(turretPhysEid);
    setNodeRender(gunNode, gunRenderEid);
    linkBrainChild(turretBrain, gunNode);

    return [gunPhysEid, gunRenderEid, gunPid];
}
