import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { getPhysicsWorldComponents, PhysicsWorld } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { spawnRectangleCarrier, SpawnCtx } from '../spawnPart.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleOptions } from './Options.ts';
import { spawnSoundAtParent } from '../Sound.ts';
import { SoundType } from '../../Components/Sound.ts';

const volumeByType: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 0.6,
    [VehicleType.MediumTank]: 0.8,
    [VehicleType.HeavyTank]: 1.0,
    [VehicleType.PlayerTank]: 0.8,
    [VehicleType.Harvester]: 1.0,
    [VehicleType.MeleeCar]: 0.7,
};

// Returns [vehiclePhysEid, vehicleRenderEid, vehiclePid]
export function createVehicleBase(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: VehicleOptions,
): [number, number, number] {
    const {
        Vehicle, TeamRef, PlayerRef, LastHitters,
        HeuristicsData, ImpulseAtPoint, VehicleController,
    } = getPhysicsWorldComponents(world);
    const renderWorld = Worlds.renderWorld;
    const { Color, Children, SoundParentRelative } = getRenderWorldComponents(renderWorld);

    options.belongsCollisionGroup = CollisionGroup.VEHICALE_BASE;
    options.interactsCollisionGroup = CollisionGroup.VEHICALE_BASE;

    const ctx: SpawnCtx = { physicsWorld: world, renderWorld, physicalWorld };
    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = spawnRectangleCarrier(ctx, options);

    Vehicle.addComponent(world, vehiclePhysEid, options.vehicleType);
    Vehicle.setEngineType(vehiclePhysEid, options.engineType);
    TeamRef.addComponent(world, vehiclePhysEid, options.teamId);
    PlayerRef.addComponent(world, vehiclePhysEid, options.playerId);
    LastHitters.addComponent(world, vehiclePhysEid);
    HeuristicsData.addComponent(world, vehiclePhysEid, options.approximateColliderRadius);
    ImpulseAtPoint.addComponent(world, vehiclePhysEid);
    VehicleController.addComponent(world, vehiclePhysEid);

    // Render hierarchy on the mirror.
    Children.addComponent(renderWorld, vehicleRenderEid);
    Color.addComponent(renderWorld, vehicleRenderEid, ...options.color);

    const soundEid = spawnSoundAtParent(renderWorld, {
        parentEid: vehicleRenderEid,
        type: SoundType.TankMove,
        volume: volumeByType[options.vehicleType] ?? 0.8,
        loop: true,
        autoplay: false,
    });
    SoundParentRelative.addComponent(renderWorld, soundEid);

    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

export type TurretOptions = {
    rotationSpeed: number;
};

// Returns [turretPhysEid, turretRenderEid, turretPid]
export function createVehicleTurret(
    world: PhysicsWorld,
    physicalWorld: PhysicalWorld,
    options: VehicleOptions,
    turretOptions: TurretOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
): [number, number, number] {
    const { VehicleTurret, TurretController, Joint, JointMotor } = getPhysicsWorldComponents(world);
    const renderWorld = Worlds.renderWorld;
    const { Parent, Children } = getRenderWorldComponents(renderWorld);

    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;

    const ctx: SpawnCtx = { physicsWorld: world, renderWorld, physicalWorld };
    const [turretPhysEid, turretRenderEid, turretPid] = spawnRectangleCarrier(ctx, options);

    VehicleTurret.addComponent(world, turretPhysEid);
    VehicleTurret.setRotationSpeed(turretPhysEid, turretOptions.rotationSpeed);
    TurretController.addComponent(world, turretPhysEid);

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
    Joint.addComponent(world, turretPhysEid, joint.handle);
    JointMotor.addComponent(world, turretPhysEid);

    Parent.addComponent(renderWorld, turretRenderEid, vehicleRenderEid);
    Children.addComponent(renderWorld, turretRenderEid);
    Children.addChildren(vehicleRenderEid, turretRenderEid);

    return [turretPhysEid, turretRenderEid, turretPid];
}
