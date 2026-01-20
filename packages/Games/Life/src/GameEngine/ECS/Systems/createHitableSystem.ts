import { GameDI } from '../../DI/GameDI.js';
import { Hitable } from '../Components/Hitable.js';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.js';
import { EntityId, hasComponent, Not, onSet, query } from 'bitecs';
import { Bullet, BulletCaliber, mapBulletCaliber } from '../Components/Bullet.js';
import { createChangeDetector } from '../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { VehiclePart } from '../Components/VehiclePart.js';
import { getTankHealth, tearOffTankPart } from '../Entities/Tank/TankUtils.js';
import { Score } from '../Components/Score.js';
import { TeamRef } from '../Components/TeamRef.js';
import { PlayerRef } from '../Components/PlayerRef.js';
import { Damagable } from '../Components/Damagable.js';
import { spawnHitFlash } from '../Entities/HitFlash.js';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { clamp } from 'lodash';
import { Parent } from '../Components/Parent.js';
import { Vehicle } from '../Components/Vehicle.js';
import { SoundType } from '../Components/Sound.js';
import { spawnSoundAtParent } from '../Entities/Sound.js';
import { LastHitters } from '../Components/LastHitters.js';
import { WEIGHTS } from '../../../../../ml/src/Reward/calculateReward.ts';

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
            saveHitters(vehiclePartEid, vehicleEid, hitEids);

            if (hasComponent(world, vehicleEid, Vehicle)) {
                hittedVehicles.add(vehicleEid);
            }
            
            if (!Hitable.isDestroyed(vehiclePartEid)) continue;

            tearOffTankPart(vehiclePartEid, true);
            
            if (getTankHealth(vehicleEid) === 0) {
                awardKillReward(vehicleEid);
            }
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
    hittableEid: EntityId, // entity that was hit (vehicle part)
    hitEids: Float64Array, // entities that hit the hittable
    { world } = GameDI,
) {
    for (const hitEid of hitEids) {
        if (!hasComponent(world, hitEid, PlayerRef)) continue;
        if (!hasComponent(world, hitEid, TeamRef)) continue;
        if (!hasComponent(world, hittableEid, TeamRef)) continue;
        
        const attackerPlayerId = PlayerRef.id[hitEid];
        const vehiclePartTeamId = TeamRef.id[hittableEid];
        const attackerTeamId = TeamRef.id[hitEid];
        const isFriendlyFire = vehiclePartTeamId === attackerTeamId;

        // Reduced hit reward (shifted focus to kills)
        Score.updateScore(attackerPlayerId, isFriendlyFire ? (-2 * WEIGHTS.HIT_REWARD) : WEIGHTS.HIT_REWARD);
    }

    // Penalize the hittable for being hit (trade efficiency)
    const hittablePlayerId = PlayerRef.id[hittableEid];
    Score.updateScore(hittablePlayerId, -(0.5 * WEIGHTS.HIT_REWARD * hitEids.length));
}

function saveHitters(
    hittableEid: EntityId, // entity that was hit (vehicle part)
    vehicleEid: EntityId, // vehicle that owns this part
    hitEids: Float64Array, // entities that hit the hittable
    { world } = GameDI,
) {
    if (!hasComponent(world, vehicleEid, LastHitters)) return;
    if (!hasComponent(world, hittableEid, TeamRef)) return;
    
    const vehiclePartTeamId = TeamRef.id[hittableEid];
    
    for (const hitEid of hitEids) {
        if (!hasComponent(world, hitEid, PlayerRef)) continue;
        if (!hasComponent(world, hitEid, TeamRef)) continue;
        
        const attackerTeamId = TeamRef.id[hitEid];
        if (attackerTeamId === vehiclePartTeamId) continue; // Skip friendly fire
        
        const attackerPlayerId = PlayerRef.id[hitEid];
        LastHitters.addHitter(vehicleEid, attackerPlayerId);
    }
}

function awardKillReward(destroyedVehicleEid: EntityId, { world } = GameDI) {
    if (!hasComponent(world, destroyedVehicleEid, LastHitters)) return;
    
    const hitters = LastHitters.getAllHitters(destroyedVehicleEid);
    
    for (const playerId of hitters) {
        Score.updateScore(playerId, WEIGHTS.KILL_REWARD);
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
