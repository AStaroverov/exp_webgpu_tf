import { GameDI } from '../../DI/GameDI.ts';
import { Hitable } from '../Components/Hitable.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { EntityId, hasComponent, onSet, query } from 'bitecs';
import { Bullet, BulletCaliber, mapBulletCaliber } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { TankPart } from '../Components/TankPart.ts';
import { tearOffTankPart } from '../Entities/Tank/TankUtils.ts';
import { Score } from '../Components/Score.ts';
import { TeamRef } from '../Components/TeamRef.ts';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { Damagable } from '../Components/Damagable.ts';
import { min } from '../../../../../../lib/math.ts';
import { spawnHitFlash } from '../Entities/HitFlash.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';

export function createHitableSystem({ world } = GameDI) {
    const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);

    return () => {
        if (!hitableChanges.hasChanges()) return;

        const tankPartEids = query(world, [TankPart, Hitable]);
        for (let i = 0; i < tankPartEids.length; i++) {
            const tankPartEid = tankPartEids[i];
            if (!hitableChanges.has(tankPartEid)) continue;

            const hitEids = Hitable.getHitEids(tankPartEid);

            applyDamage(tankPartEid);
            applyScores(tankPartEid, hitEids);

            if (!Hitable.isDestroyed(tankPartEid)) continue;

            tearOffTankPart(tankPartEid, true);
        }

        const bulletIds = query(world, [Bullet, Hitable]);
        for (let i = 0; i < bulletIds.length; i++) {
            const bulletId = bulletIds[i];
            if (!hitableChanges.has(bulletId)) continue;

            applyDamage(bulletId);

            if (!Hitable.isDestroyed(bulletId)) continue;

            // Spawn hit flash at bullet position
            const bulletMatrix = GlobalTransform.matrix.getBatch(bulletId);
            const bulletCaliber = mapBulletCaliber[Bullet.caliber[bulletId] as BulletCaliber];
            spawnHitFlash({
                x: getMatrixTranslationX(bulletMatrix),
                y: getMatrixTranslationY(bulletMatrix),
                size: bulletCaliber.width * 2,
                duration: 400,
            });

            scheduleRemoveEntity(bulletId);
        }

        hitableChanges.clear();
    };
}

const FORCE_TARGET = 10_000_000;

function applyDamage(targetEid: number, { world } = GameDI) {
    const count = Hitable.hitIndex[targetEid];
    const hits = Hitable.hits.getBatch(targetEid);

    for (let i = 0; i < count; i++) {
        const sourceEid = hits[i * 2];
        const forceCoeff = min(1, hits[i * 2 + 1] / FORCE_TARGET);
        const damage = hasComponent(world, sourceEid, Damagable)
            ? forceCoeff * Damagable.damage[sourceEid]
            : 0;

        Hitable.health[targetEid] -= damage;
    }
    Hitable.resetHits(targetEid);
}

function applyScores(tankPartEid: EntityId, hitEids: Float64Array, { world } = GameDI) {
    for (const hitEid of hitEids) {
        if (
            hasComponent(world, tankPartEid, TeamRef)
            && hasComponent(world, hitEid, TeamRef)
            && hasComponent(world, hitEid, PlayerRef)
        ) {
            const tankPartTeamId = TeamRef.id[tankPartEid];
            const secondTeamId = TeamRef.id[hitEid];
            const playerId = PlayerRef.id[hitEid];

            Score.updateScore(playerId, tankPartTeamId === secondTeamId ? -1 : 1);
        }
    }

}