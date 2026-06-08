import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { query } from 'bitecs';
import { getGameComponents } from '../../createGameWorld.ts';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { MapDI } from '../../../DI/MapDI.ts';
import { SpottingConfig } from '../../../Config/index.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { createRectangleRigidGroup } from '../../Components/RigidGroup.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleOptions } from './Options.ts';
import { spawnSoundAtParent } from '../Sound.ts';
import { SoundType } from '../../Components/Sound.ts';

const volumeByType: Record<VehicleType, number> = {
    [VehicleType.LightTank]: 0.6,
    [VehicleType.MediumTank]: 0.8,
    [VehicleType.HeavyTank]: 1.0,
    [VehicleType.Ranger]: 0.6,
    [VehicleType.Harvester]: 1.0,
    [VehicleType.MeleeCar]: 0.7,
};

export function createVehicleBase(options: VehicleOptions, { world } = GameDI): [number, number] {
    const {
        Vehicle, Color, Children, TeamRef, PlayerRef, LastHitters,
        HeuristicsData, ImpulseAtPoint, VehicleController, SoundParentRelative, Spottable,
    } = getGameComponents(world);

    options.belongsCollisionGroup = CollisionGroup.VEHICALE_BASE;
    options.interactsCollisionGroup = CollisionGroup.VEHICALE_BASE;

    const [vehicleEid, vehiclePid] = createRectangleRigidGroup(options);
    Vehicle.addComponent(world, vehicleEid, options.vehicleType);
    Vehicle.setEngineType(vehicleEid, options.engineType);

    addTransformComponents(world, vehicleEid);
    Children.addComponent(world, vehicleEid);
    Color.addComponent(world, vehicleEid, ...options.color);
    TeamRef.addComponent(world, vehicleEid, options.teamId);
    Spottable.addComponent(world, vehicleEid);
    PlayerRef.addComponent(world, vehicleEid, options.playerId);
    LastHitters.addComponent(world, vehicleEid);
    HeuristicsData.addComponent(world, vehicleEid, options.approximateColliderRadius);

    ImpulseAtPoint.addComponent(world, vehicleEid);

    VehicleController.addComponent(world, vehicleEid);

    const soundEid = spawnSoundAtParent({
        parentEid: vehicleEid,
        type: SoundType.TankMove,
        volume: volumeByType[options.vehicleType] ?? 0.8,
        loop: true,
        autoplay: false,
    });
    SoundParentRelative.addComponent(world, soundEid);

    return [vehicleEid, vehiclePid];
}

const parentVector = new Vector2(0, 0);
const childVector = new Vector2(0, 0);

export type TurretOptions = {
    rotationSpeed: number;
};

export function createVehicleTurret(
    options: VehicleOptions,
    turretOptions: TurretOptions,
    vehicleEid: number,
    vehiclePid: number,
    { world, physicalWorld } = GameDI,
): [number, number] {
    const { VehicleTurret, TurretController, Joint, JointMotor, Parent, Children } = getGameComponents(world);

    options.belongsCollisionGroup = 0;
    options.interactsCollisionGroup = 0;

    const [turretEid, turretPid] = createRectangleRigidGroup(options);
    VehicleTurret.addComponent(world, turretEid, turretOptions.rotationSpeed);
    TurretController.addComponent(world, turretEid);

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
    Joint.addComponent(world, turretEid, joint.handle);
    JointMotor.addComponent(world, turretEid);

    addTransformComponents(world, turretEid);
    Parent.addComponent(world, turretEid, vehicleEid);
    Children.addComponent(world, turretEid);
    Children.addChildren(vehicleEid, turretEid);

    return [turretEid, turretPid];
}

/**
 * Reveal on firing — a one-shot confidence impulse on the shooter vehicle.
 *
 * When a vehicle fires, if any opposing unit is within `< SpottingConfig.fireRevealDist`
 * cells (hex steps) of the shooter, he gets a confidence impulse (`markSpotted` to the
 * `fire` level): the muzzle flash gives his rough position away. This runs late in the
 * tick (spawnFrame), so the spotting system decays it from that level next tick — a
 * medium, fading reveal. Vehicle entity method — reads its own position/team and writes
 * its own `Spottable`; the opposing-unit scan only decides whether to raise it.
 *
 * Fresh per-shot query over vehicles (few units, cheap). No line-of-sight check.
 */
export function revealByFire(shooterVehicleEid: number, { world } = GameDI): void {
    const grid = MapDI.grid;
    if (!grid) return;

    const { Vehicle, TeamRef, RigidBodyState, Spottable } = getGameComponents(world);

    const shooterHex = grid.worldToHex(
        RigidBodyState.position.get(shooterVehicleEid, 0),
        RigidBodyState.position.get(shooterVehicleEid, 1),
    );
    if (!shooterHex) return;

    const shooterTeam = TeamRef.id[shooterVehicleEid];

    // Already revealed at least as strongly (beam / fresh fire) — nothing to add.
    if (Spottable.getConfidence(shooterVehicleEid) >= SpottingConfig.confidence.fire) return;

    const vehicles = query(world, [Vehicle, TeamRef, RigidBodyState, Spottable]);
    for (const eid of vehicles) {
        if (TeamRef.id[eid] === shooterTeam) continue;

        const hex = grid.worldToHex(
            RigidBodyState.position.get(eid, 0),
            RigidBodyState.position.get(eid, 1),
        );
        if (!hex) continue;

        if (grid.distance(shooterHex, hex) < SpottingConfig.fireRevealDist) {
            Spottable.markSpotted(shooterVehicleEid, SpottingConfig.confidence.fire);
            return; // one opposing unit in range is enough
        }
    }
}
