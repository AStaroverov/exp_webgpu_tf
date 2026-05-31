import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { PhysicalWorld } from '../../../../Physical/initPhysicalWorld.ts';
import { TrackSide } from '../../../Components/Track.ts';
import { createVehicleBase, createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { createTrack, TrackOptions } from '../../Track/createTrack.ts';
import { type TankOptions } from './Options.ts';
import { spawnRectangleCarrier, SpawnCtx } from '../../spawnPart.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../../createRenderWorld.ts';
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
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: TankOptions,
): [number, number, number] {
    const { Tank } = getPhysicsWorldComponents(world);
    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = createVehicleBase(world, physicalWorld, options);
    Tank.addComponent(world, vehiclePhysEid);
    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

// Returns [leftTrackRenderEid, rightTrackRenderEid]
export function createTankTracks(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
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
    const [, leftTrackRenderEid] = createTrack(world, physicalWorld, trackOptions, tankRenderEid, tankPid);

    trackOptions.trackSide = TrackSide.Right;
    trackOptions.anchorY = tracksConfig.rightAnchorY;
    trackOptions.x = options.x;
    trackOptions.y = options.y;
    const [, rightTrackRenderEid] = createTrack(world, physicalWorld, trackOptions, tankRenderEid, tankPid);

    return [leftTrackRenderEid, rightTrackRenderEid];
}

// Returns [turretRenderEid, gunRenderEid]
export function createTankTurret(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: TankOptions,
    tankPhysEid: number,
    tankRenderEid: number,
    tankPid: number,
): readonly [number, number] {
    const { Tank, Firearms } = getPhysicsWorldComponents(world);

    const [turretPhysEid, turretRenderEid, turretPid] = createVehicleTurret(
        world,
        physicalWorld,
        options,
        options.turret,
        tankRenderEid,
        tankPid,
    );

    const [, gunRenderEid] = createTankGun(world, physicalWorld, options, turretRenderEid, turretPid);

    Tank.setTurretEid(tankPhysEid, turretPhysEid);

    Firearms.addComponent(world, turretPhysEid);
    Firearms.setData(turretPhysEid, options.firearms.bulletStartPosition, options.firearms.bulletCaliber);
    Firearms.setReloadingDuration(turretPhysEid, options.firearms.reloadingDuration);

    return [turretRenderEid, gunRenderEid] as const;
}

// Returns [gunPhysEid, gunRenderEid, gunPid]
export function createTankGun(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: TankOptions,
    turretRenderEid: number,
    turretPid: number,
): [number, number, number] {
    const { Joint } = getPhysicsWorldComponents(world);
    const renderWorld = Worlds.renderWorld;
    const { Parent, Children } = getRenderWorldComponents(renderWorld);

    const ctx: SpawnCtx = { physicsWorld: world, renderWorld, physicalWorld };
    const [gunPhysEid, gunRenderEid, gunPid] = spawnRectangleCarrier(ctx, {
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
    Joint.addComponent(world, gunPhysEid, joint.handle);

    Parent.addComponent(renderWorld, gunRenderEid, turretRenderEid);
    Children.addComponent(renderWorld, gunRenderEid);
    Children.addChildren(turretRenderEid, gunRenderEid);

    return [gunPhysEid, gunRenderEid, gunPid];
}
