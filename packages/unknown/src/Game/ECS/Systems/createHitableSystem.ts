import { Worlds } from '../../DI/Worlds.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { EntityId, hasComponent, Not, onSet, query } from 'bitecs';
import { BulletCaliber, mapBulletCaliber } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { getTankHealth, tearOffTankPart } from '../Entities/Tank/TankUtils.ts';
import { spawnHitFlash } from '../Entities/HitFlash.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { clamp } from 'lodash';
import { SoundType } from '../Components/Sound.ts';
import { spawnSoundAtParent } from '../Entities/Sound.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';

export function createHitableSystem({ physicsWorld, renderWorld } = Worlds) {
    const { Hitable, Bullet, VehiclePart, Vehicle } = getPhysicsWorldComponents(physicsWorld);
    const hitableChanges = createChangeDetector(physicsWorld, [onSet(Hitable)]);
    let time = 0;

    return (delta: number) => {
        const { Parent } = getRenderWorldComponents(renderWorld);
        time += delta;

        if (!hitableChanges.hasChanges()) return;

        const vehiclePartEids = query(physicsWorld, [VehiclePart, Hitable]);
        const hittedVehicles = new Set<EntityId>();

        for (let i = 0; i < vehiclePartEids.length; i++) {
            const vehiclePartEid = vehiclePartEids[i];
            if (!hitableChanges.has(vehiclePartEid)) continue;

            const hitEids = Hitable.getHitEids(vehiclePartEid);
            // part phys -> part render -> slot render -> carrier render -> carrier phys (vehicle)
            const partRenderEid = BridgeDI.getRenderOf(vehiclePartEid);
            const slotRenderEid = Parent.id[partRenderEid];
            const carrierRenderEid = Parent.id[slotRenderEid];
            const vehicleEid = BridgeDI.getPhysicsOf(carrierRenderEid);

            applyDamage(physicsWorld, vehiclePartEid);
            saveHitters(physicsWorld, vehiclePartEid, vehicleEid, hitEids);

            if (hasComponent(physicsWorld, vehicleEid, Vehicle)) {
                hittedVehicles.add(vehicleEid);
            }

            if (!Hitable.isDestroyed(vehiclePartEid)) continue;

            tearOffTankPart(vehiclePartEid, true);

            getTankHealth(vehicleEid);
        }

        for (const vehicleEid of hittedVehicles) {
           throttledSpawnSoundAtParent(renderWorld, BridgeDI.getRenderOf(vehicleEid), time, 200);
        }

        const bulletIds = query(physicsWorld, [Bullet, Hitable]);
        for (let i = 0; i < bulletIds.length; i++) {
            const bulletId = bulletIds[i];
            if (!hitableChanges.has(bulletId)) continue;

            applyDamage(physicsWorld, bulletId);

            if (!Hitable.isDestroyed(bulletId)) continue;

            const bulletMatrix = GlobalTransform.matrix.getBatch(BridgeDI.getRenderOf(bulletId));
            const bulletCaliber = mapBulletCaliber[Bullet.caliber[bulletId] as BulletCaliber];
            const hitX = getMatrixTranslationX(bulletMatrix);
            const hitY = getMatrixTranslationY(bulletMatrix);

            spawnHitFlash(renderWorld, {
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
    hittableEid: EntityId,
    vehicleEid: EntityId,
    hitEids: Float64Array,
) {
    const { LastHitters, TeamRef, PlayerRef } = getPhysicsWorldComponents(world);
    if (!hasComponent(world, vehicleEid, LastHitters)) return;
    if (!hasComponent(world, hittableEid, TeamRef)) return;

    const vehiclePartTeamId = TeamRef.id[hittableEid];

    for (const hitEid of hitEids) {
        if (!hasComponent(world, hitEid, PlayerRef)) continue;
        if (!hasComponent(world, hitEid, TeamRef)) continue;

        const attackerTeamId = TeamRef.id[hitEid];
        if (attackerTeamId === vehiclePartTeamId) continue;

        const attackerPlayerId = PlayerRef.id[hitEid];
        LastHitters.addHit(vehicleEid, attackerPlayerId);
    }
}

const mapParentToLastSoundTime = new Map<EntityId, number>();
function throttledSpawnSoundAtParent(world: RenderGameWorld, parentEid: EntityId, now: number, delay: number) {
    const lastSpawnTime = mapParentToLastSoundTime.get(parentEid);
    if (lastSpawnTime && (now - lastSpawnTime) < delay) return;
    mapParentToLastSoundTime.set(parentEid, now);

    spawnSoundAtParent(world, {
        parentEid,
        type: SoundType.TankHit,
        loop: false,
        volume: 1,
        autoplay: true,
    });
}
