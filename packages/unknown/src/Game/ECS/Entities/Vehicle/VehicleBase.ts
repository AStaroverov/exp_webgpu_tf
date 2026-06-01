import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { addEntity } from 'bitecs';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { spawnRectangleCarrier } from '../spawnPart.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { getNodeByPhysics, getPhysicsOf, setNodeRender, linkBrainChild } from '../../refs.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleOptions } from './Options.ts';
import { spawnSoundForOwner } from '../Sound.ts';
import { SoundType } from '../../Components/Sound.ts';
import { getSoundWorldComponents } from '../../createSoundWorld.ts';

const volumeByType: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 0.6,
    [VehicleType.MediumTank]: 0.8,
    [VehicleType.HeavyTank]: 1.0,
    [VehicleType.PlayerTank]: 0.8,
    [VehicleType.Harvester]: 1.0,
    [VehicleType.MeleeCar]: 0.7,
};

// Resolves the hull node (= hull-brain) given the vehicle's RENDER eid: render ->
// physics atom -> the brain node whose presentation is that atom (downward refs only).
export function getHullBrainByRender(vehicleRenderEid: number): number {
    return getNodeByPhysics(getPhysicsOf(vehicleRenderEid));
}

// Returns [vehiclePhysEid, vehicleRenderEid, vehiclePid]
export function createVehicleBase(
    options: VehicleOptions,
    { physicsWorld, brainWorld, renderWorld, soundWorld } = Worlds,
): [number, number, number] {
    const { ImpulseAtPoint } = getPhysicsWorldComponents(physicsWorld);
    const {
        Vehicle, TeamRef, PlayerRef, LastHitters, HeuristicsData, VehicleController,
    } = getBrainWorldComponents(brainWorld);
    const { Color, Children } = getRenderWorldComponents(renderWorld);
    const { SoundParentRelative } = getSoundWorldComponents(soundWorld);

    options.belongsCollisionGroup = CollisionGroup.VEHICALE_BASE;
    options.interactsCollisionGroup = CollisionGroup.VEHICALE_BASE;

    // Brain-first: lay the hull node out before its physics/render presentation.
    const hullBrain = addEntity(brainWorld);

    const [vehiclePhysEid, vehicleRenderEid, vehiclePid] = spawnRectangleCarrier(options);

    // The hull-brain: the vehicle's mind. Canonical team/player live here.
    Vehicle.addComponent(brainWorld, hullBrain, options.vehicleType);
    Vehicle.setEngineType(hullBrain, options.engineType);
    TeamRef.addComponent(brainWorld, hullBrain, options.teamId);
    PlayerRef.addComponent(brainWorld, hullBrain, options.playerId);
    LastHitters.addComponent(brainWorld, hullBrain);
    HeuristicsData.addComponent(brainWorld, hullBrain, options.approximateColliderRadius);
    VehicleController.addComponent(brainWorld, hullBrain);

    // Per-atom physics impulse helper stays on the atom.
    ImpulseAtPoint.addComponent(physicsWorld, vehiclePhysEid);

    // Node->render presentation (scheme A: drawn node carries NodeRenderRef; the
    // render's PhysicsRef -> physics already set in spawnRectangleCarrier). The node
    // reaches its hull atom downward via getNodePhysics(hullBrain).
    setNodeRender(hullBrain, vehicleRenderEid);

    // Render hierarchy on the mirror.
    Children.addComponent(renderWorld, vehicleRenderEid);
    Color.addComponent(renderWorld, vehicleRenderEid, ...options.color);

    const soundEid = spawnSoundForOwner({
        ownerEid: vehiclePhysEid,
        type: SoundType.TankMove,
        volume: volumeByType[options.vehicleType] ?? 0.8,
        loop: true,
        autoplay: false,
    });
    SoundParentRelative.addComponent(soundWorld, soundEid);

    return [vehiclePhysEid, vehicleRenderEid, vehiclePid];
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

export type TurretOptions = {
    rotationSpeed: number;
};

// Returns [turretPhysEid, turretRenderEid, turretPid]
export function createVehicleTurret(
    options: VehicleOptions,
    turretOptions: TurretOptions,
    vehicleRenderEid: number,
    vehiclePid: number,
    { physicsWorld, physicalWorld, brainWorld } = Worlds,
): [number, number, number] {
    const { VehicleTurret, Joint, JointMotor } = getPhysicsWorldComponents(physicsWorld);
    const { TurretController } = getBrainWorldComponents(brainWorld);

    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;

    // Brain-first: the turret node before its physics/render presentation.
    const turretBrain = addEntity(brainWorld);

    const [turretPhysEid, turretRenderEid, turretPid] = spawnRectangleCarrier(options);

    VehicleTurret.addComponent(physicsWorld, turretPhysEid);
    VehicleTurret.setRotationSpeed(turretPhysEid, turretOptions.rotationSpeed);

    // The turret-brain: aim/fire mind, distinct from the hull-brain.
    TurretController.addComponent(brainWorld, turretBrain);

    // Node->render presentation + Brain hierarchy: turret node is a child of the hull
    // node; it reaches its turret atom downward via getNodePhysics(turretBrain).
    setNodeRender(turretBrain, turretRenderEid);
    linkBrainChild(getHullBrainByRender(vehicleRenderEid), turretBrain);

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
    Joint.addComponent(physicsWorld, turretPhysEid, joint.handle);
    JointMotor.addComponent(physicsWorld, turretPhysEid);

    // No render parent/children: the turret carrier is physics-driven — its mirror gets
    // a WORLD transform from mirrorSync, so it must be a render ROOT (Global = world).
    // Parenting it under the hull would double-compose (hull.Global x turret.Local=world).

    return [turretPhysEid, turretRenderEid, turretPid];
}
