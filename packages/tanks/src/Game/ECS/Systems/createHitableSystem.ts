import { GameDI } from '../../DI/GameDI.ts';
import { Hitable } from '../Components/Hitable.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { EntityId, hasComponent, Not, onSet, query } from 'bitecs';
import { Bullet, BulletCaliber, mapBulletCaliber } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { VehiclePart } from '../Components/VehiclePart.ts';
import { tearOffTankPart } from '../Entities/Tank/TankUtils.ts';
import { Score } from '../Components/Score.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { Damagable } from '../Components/Damagable.ts';
import { spawnHitFlash } from '../Entities/HitFlash.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { clamp } from 'lodash';
import { Parent } from '../Components/Parent.ts';
import { Vehicle } from '../Components/Vehicle.ts';
import { SoundType } from '../Components/Sound.ts';
import { spawnSoundAtParent } from '../Entities/Sound.ts';
import { Obstacle } from '../Components/Obstacle.ts';

export function createHitableSystem({ world } = GameDI) {
    const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);
    let time = 0;

    return (delta: number) => {
        time += delta;

        if (!hitableChanges.hasChanges()) return;

        const vehiclePartEids = query(world, [VehiclePart, Hitable]);
        const hittedVehicles = new Set<EntityId>();
        
        for (let i = 0; i < vehiclePartEids.length; i++) {
            const vehiclePartEid = vehiclePartEids[i];
            if (!hitableChanges.has(vehiclePartEid)) continue;

            const hitEids = Hitable.getHitEids(vehiclePartEid);
            const vehicleEid = Parent.id[Parent.id[vehiclePartEid]];

            applyDamage(vehiclePartEid);
            applyScores(vehiclePartEid, hitEids);

            if (hasComponent(world, vehicleEid, Vehicle)) {
                hittedVehicles.add(vehicleEid);
            }
            
            if (!Hitable.isDestroyed(vehiclePartEid)) continue;

            tearOffTankPart(vehiclePartEid, true);
        }

        for (const vehicleEid of hittedVehicles) {
           throttledSpawnSoundAtParent(vehicleEid, time, 200);
        }

        const bulletIds = query(world, [Bullet, Hitable]);
        for (let i = 0; i < bulletIds.length; i++) {
            const bulletId = bulletIds[i];
            if (!hitableChanges.has(bulletId)) continue;

            applyDamage(bulletId);

            if (!Hitable.isDestroyed(bulletId)) continue;

            // Spawn hit flash at bullet position (includes hit sound)
            const bulletMatrix = GlobalTransform.matrix.getBatch(bulletId);
            const bulletCaliber = mapBulletCaliber[Bullet.caliber[bulletId] as BulletCaliber];
            const hitX = getMatrixTranslationX(bulletMatrix);
            const hitY = getMatrixTranslationY(bulletMatrix);
         
            spawnHitFlash({
                x: hitX,
                y: hitY,
                size: bulletCaliber.width * 2,
                duration: 400,
            });

            scheduleRemoveEntity(bulletId);
        }

        const restEids = query(world, [Hitable, Not(VehiclePart), Not(Bullet)]);
        for (let i = 0; i < restEids.length; i++) {
            const eid = restEids[i];
            if (!hitableChanges.has(eid)) continue;

            const hitEids = Hitable.getHitEids(eid);
            applyScores(eid, hitEids);
            applyDamage(eid);
            
            if (!Hitable.isDestroyed(eid)) continue;

            scheduleRemoveEntity(eid);
        }

        hitableChanges.clear();
    };
}

const FORCE_TARGET = 1_000_000_000;

function applyDamage(targetEid: number, { world } = GameDI) {
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

function applyScores(
    hittableEid: EntityId, // entity that was hit
    hitEids: Float64Array, // entities that hit the hittable
    { world } = GameDI,
) {
    for (const hitEid of hitEids) {
        if (!hasComponent(world, hitEid, PlayerRef)) continue;
        
        const playerId = PlayerRef.id[hitEid];
        
        if (hasComponent(world, hittableEid, Obstacle)) {
            Score.updateScore(playerId, -0.3); // penalty for hitting an obstacle, just for ML
        } else if (hasComponent(world, hitEid, TeamRef) && hasComponent(world, hittableEid, TeamRef)) {
            const vehiclePartTeamId = TeamRef.id[hittableEid];
            const secondTeamId = TeamRef.id[hitEid];

            Score.updateScore(playerId, vehiclePartTeamId === secondTeamId ? -1 : 1);
        }
    }

}

const mapParentToLastSoundTime = new Map<EntityId, number>();
function throttledSpawnSoundAtParent(parentEid: EntityId, now: number, delay: number) {
    const lastSpawnTime = mapParentToLastSoundTime.get(parentEid);
    if (lastSpawnTime && (now - lastSpawnTime) < delay) return;
    mapParentToLastSoundTime.set(parentEid, now);

    spawnSoundAtParent({
        parentEid,
        type: SoundType.TankHit,
        loop: false,
        volume: 1,
        autoplay: true,
    });
}