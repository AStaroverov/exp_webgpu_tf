import { GameDI } from '../../DI/GameDI.ts';
import { Hitable } from '../Components/Hitable.ts';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.ts';
import { hasComponent, onSet, query } from 'bitecs';
import { Bullet } from '../Components/Bullet.ts';
import { createChangeDetector } from '../../../../../src/ECS/Systems/ChangedDetectorSystem.ts';
import { TankPart } from '../Components/TankPart.ts';
import { tearOffTankPart } from '../Entities/Tank/TankUtils.ts';
import { Score } from '../Components/Score.ts';
import { TeamRef } from '../Components/TeamRef.ts';

export function createHitableSystem({ world } = GameDI) {
    const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);

    return () => {
        if (!hitableChanges.hasChanges()) return;

        const tankPartEids = query(world, [TankPart, Hitable]);
        for (let i = 0; i < tankPartEids.length; i++) {
            const tankPartEid = tankPartEids[i];
            if (!hitableChanges.has(tankPartEid)) continue;

            const damage = Hitable.damage[tankPartEid];
            if (damage <= 0) continue;

            tearOffTankPart(tankPartEid);

            const secondEid = Hitable.secondEid[tankPartEid];
            if (!hasComponent(world, tankPartEid, TeamRef) || !hasComponent(world, secondEid, TeamRef)) continue;

            const tankPartTeamId = TeamRef.id[tankPartEid];
            const secondTeamId = TeamRef.id[secondEid];

            Score.updateScore(secondTeamId, tankPartTeamId === secondTeamId ? -1 : 1);
        }

        const bulletIds = query(world, [Bullet, Hitable]);
        for (let i = 0; i < bulletIds.length; i++) {
            const bulletId = bulletIds[i];
            if (!hitableChanges.has(bulletId)) continue;

            const damage = Hitable.damage[bulletId];
            if (damage <= 0) continue;

            scheduleRemoveEntity(bulletId);
        }

        hitableChanges.clear();
    };
}