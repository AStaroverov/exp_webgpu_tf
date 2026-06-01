import { Worlds } from '../../DI/Worlds.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { EntityId, hasComponent, Not, onSet, query } from 'bitecs';
import { BulletCaliber, mapBulletCaliber } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { getTankHealth, tearOffTankPart } from '../Entities/Tank/TankUtils.ts';
import { spawnHitFlash } from '../Entities/HitFlash.ts';
import { clamp } from 'lodash';
import { SoundType } from '../Components/Sound.ts';
import { spawnSoundForOwner } from '../Entities/Sound.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';
import { getBrainWorldComponents, BrainWorld } from '../createBrainWorld.ts';
import { getCarrierNodeOfPart, getNodeParent, getNodePhysics } from '../refs.ts';

export function createHitableSystem({ physicsWorld, brainWorld } = Worlds) {
    const { Hitable, Bullet, VehiclePart, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);
    const { Vehicle, TurretController } = getBrainWorldComponents(brainWorld);
    const hitableChanges = createChangeDetector(physicsWorld, [onSet(Hitable)]);
    let time = 0;

    return (delta: number) => {
        time += delta;

        if (!hitableChanges.hasChanges()) return;

        const vehiclePartEids = query(physicsWorld, [VehiclePart, Hitable]);
        const hittedVehicleBrains = new Set<EntityId>();

        for (let i = 0; i < vehiclePartEids.length; i++) {
            const vehiclePartEid = vehiclePartEids[i];
            if (!hitableChanges.has(vehiclePartEid)) continue;

            const hitEids = Hitable.getHitEids(vehiclePartEid);
            // Resolve the part's vehicle top-down from its carrier node (the node whose
            // slot holds it), reproducing the old part->carrier-atom->attached-brain
            // chain. A node owns its OWN brain only if it carries Vehicle (hull) or
            // TurretController (turret); track/wheel/gun atoms were attached to their
            // PARENT's brain. When that attached brain bears Vehicle, the credited body
            // is the carrier atom itself (hull/track); otherwise climb one node-parent.
            const [carrierNode] = getCarrierNodeOfPart(vehiclePartEid);
            const ownsBrain = hasComponent(brainWorld, carrierNode, Vehicle)
                || hasComponent(brainWorld, carrierNode, TurretController);
            const attachedBrain = ownsBrain ? carrierNode : getNodeParent(carrierNode);
            let vehicleBrain: number;
            let vehiclePhysEid: number;
            if (hasComponent(brainWorld, attachedBrain, Vehicle)) {
                vehicleBrain = attachedBrain;
                vehiclePhysEid = getNodePhysics(carrierNode);
            } else {
                vehicleBrain = getNodeParent(carrierNode);
                vehiclePhysEid = getNodePhysics(vehicleBrain);
            }

            applyDamage(physicsWorld, vehiclePartEid);
            saveHitters(physicsWorld, brainWorld, vehiclePartEid, vehicleBrain, hitEids);

            if (hasComponent(brainWorld, vehicleBrain, Vehicle)) {
                hittedVehicleBrains.add(vehiclePhysEid);
            }

            if (!Hitable.isDestroyed(vehiclePartEid)) continue;

            tearOffTankPart(vehiclePartEid, true);

            getTankHealth(vehiclePhysEid);
        }

        for (const vehiclePhysEid of hittedVehicleBrains) {
           throttledSpawnSoundForOwner(vehiclePhysEid, time, 200);
        }

        const bulletIds = query(physicsWorld, [Bullet, Hitable]);
        for (let i = 0; i < bulletIds.length; i++) {
            const bulletId = bulletIds[i];
            if (!hitableChanges.has(bulletId)) continue;

            applyDamage(physicsWorld, bulletId);

            if (!Hitable.isDestroyed(bulletId)) continue;

            // Bullets are leaf atoms (no brain node) and visually mirror their physics
            // body, so the hit point is the bullet atom's own RigidBodyState position.
            const bulletPos = RigidBodyState.position.getBatch(bulletId);
            const bulletCaliber = mapBulletCaliber[Bullet.caliber[bulletId] as BulletCaliber];
            const hitX = bulletPos[0];
            const hitY = bulletPos[1];

            spawnHitFlash({
                x: hitX,
                y: hitY,
                size: bulletCaliber.width * 2,
                duration: 400,
            });

            scheduleRemoveEntity(bulletId);
        }

        const restEids = query(physicsWorld, [Hitable, Not(VehiclePart), Not(Bullet)]);
        for (let i = 0; i < restEids.length; i++) {
            const eid = restEids[i];
            if (!hitableChanges.has(eid)) continue;

            applyDamage(physicsWorld, eid);

            if (!Hitable.isDestroyed(eid)) continue;

            scheduleRemoveEntity(eid);
        }

        hitableChanges.clear();
    };
}

const FORCE_TARGET = 1_000_000_000;

function applyDamage(world: PhysicsWorld, targetEid: number) {
    const { Hitable, Damagable } = getPhysicsWorldComponents(world);
    const count = Hitable.hitIndex[targetEid];
    const hits = Hitable.hits.getBatch(targetEid);

    for (let i = 0; i < count; i++) {
        const sourceEid = hits[i * 2];
        const forceCoeff = clamp(hits[i * 2 + 1] / FORCE_TARGET, 0, 1);
        const damage = hasComponent(world, sourceEid, Damagable)
            ? forceCoeff * Damagable.damage[sourceEid]
            : 0;

        Hitable.health[targetEid] -= damage;
    }
    Hitable.resetHits(targetEid);
}

function saveHitters(
    world: PhysicsWorld,
    brainWorld: BrainWorld,
    hittableEid: EntityId,
    vehicleBrain: EntityId,
    hitEids: Float64Array,
) {
    const { TeamRef, PlayerRef } = getPhysicsWorldComponents(world);
    const { LastHitters } = getBrainWorldComponents(brainWorld);
    // team/player live on the cheap static atom copy (atom TeamRef/PlayerRef); only the
    // LastHitters credit hops to the (hull-)brain. TeamRef/PlayerRef are co-added at
    // spawn, so TeamRef presence is the guard for "is a team/player-bearing atom".
    if (!hasComponent(brainWorld, vehicleBrain, LastHitters)) return;
    if (!hasComponent(world, hittableEid, TeamRef)) return;

    const vehiclePartTeamId = TeamRef.id[hittableEid];

    for (const hitEid of hitEids) {
        if (!hasComponent(world, hitEid, TeamRef)) continue;

        const attackerTeamId = TeamRef.id[hitEid];
        if (attackerTeamId === vehiclePartTeamId) continue;

        const attackerPlayerId = PlayerRef.id[hitEid];
        LastHitters.addHit(vehicleBrain, attackerPlayerId);
    }
}

const mapOwnerToLastSoundTime = new Map<EntityId, number>();
function throttledSpawnSoundForOwner(ownerEid: EntityId, now: number, delay: number) {
    const lastSpawnTime = mapOwnerToLastSoundTime.get(ownerEid);
    if (lastSpawnTime && (now - lastSpawnTime) < delay) return;
    mapOwnerToLastSoundTime.set(ownerEid, now);

    spawnSoundForOwner({
        ownerEid,
        type: SoundType.TankHit,
        loop: false,
        volume: 1,
        autoplay: true,
    });
}
