import { GameDI } from '../../DI/GameDI.js';
import { Hitable } from '../Components/Hitable.js';
import { scheduleRemoveEntity } from '../Utils/typicalRemoveEntity.js';
import { hasComponent, onSet, query } from 'bitecs';
import { createChangeDetector } from 'renderer/src/ECS/Systems/ChangedDetectorSystem.ts';
import { Damagable } from '../Components/Damagable.js';
import { clamp } from 'lodash';

export function createHitableSystem({ world } = GameDI) {
    const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);
    let time = 0;

    return (delta: number) => {
        time += delta;

        if (!hitableChanges.hasChanges()) return;

        const restEids = query(world, [Hitable]);
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