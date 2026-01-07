import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../../DI/GameDI.ts';
import { addTransformComponents } from '../../../../../../../renderer/src/ECS/Components/Transform.ts';
import { Firearms } from '../../../Components/Firearms.ts';
import { Tank } from '../../../Components/Tank.ts';
import { TrackSide } from '../../../Components/Track.ts';
import { createVehicleBase, createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { createTrack, TrackOptions } from '../../Track/createTrack.ts';
import { Joint } from '../../../Components/Joint.ts';
import { Parent } from '../../../Components/Parent.ts';
import { Children } from '../../../Components/Children.ts';
import { type TankOptions } from './Options.ts';
import { createRectangleRigidGroup } from '../../../Components/RigidGroup.ts';

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
) {
    const [turretEid, turretPid] = createVehicleTurret(
        options,
        options.turret,
        tankEid,
        tankPid,
    );

    const [gunEid] = createTankGun(options, turretEid, turretPid);

    // Add Tank-specific components
    Tank.setTurretEid(tankEid, turretEid);

    // Add Firearms component for shooting capability
    Firearms.addComponent(world, turretEid);
    Firearms.setData(turretEid, options.firearms.bulletStartPosition, options.firearms.bulletCaliber);
    Firearms.setReloadingDuration(turretEid, options.firearms.reloadingDuration);

    return [turretEid, gunEid] as const;
}

export function createTankGun(
    options: TankOptions,
    turretEid: number,
    turretPid: number,
    { world, physicalWorld } = GameDI,
): [number, number] {
    const [gunEid, gunPid] = createRectangleRigidGroup({
        ...options,
        width: options.turret.gunWidth,
        height: options.turret.gunHeight,
        belongsCollisionGroup: 0,
        interactsCollisionGroup: 0,
    });

    // Join gun body to turret head body
    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(
            new Vector2(options.width / 2, 0), 0,
            new Vector2(-options.turret.gunWidth / 2, 0), 0
        ),
        physicalWorld.getRigidBody(turretPid),
        physicalWorld.getRigidBody(gunPid),
        false,
    );
    Joint.addComponent(world, gunEid, joint.handle);

    // Add transform and hierarchy components for gun background
    addTransformComponents(world, gunEid);
    Parent.addComponent(world, gunEid, turretEid);
    Children.addComponent(world, gunEid);
    Children.addChildren(turretEid, gunEid);

    return [gunEid, gunPid];
}
