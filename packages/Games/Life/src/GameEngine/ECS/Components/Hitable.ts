import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from 'renderer/src/delegate.ts';
import { NestedArray, TypedArray } from 'renderer/src/utils.ts';
import { component, obs } from 'renderer/src/ECS/utils.ts';

const HITS_LIMIT = 10;

export const Hitable = component(({
    health: TypedArray.f64(delegate.defaultSize),

    hitIndex: TypedArray.i8(delegate.defaultSize),
    // @TODO: remove forceMagnitude
    // [eid, forceMagnitude]
    hits: NestedArray.f64(2 * HITS_LIMIT, delegate.defaultSize),

    addComponent: (world: World, eid: number, health: number) => {        
        addComponent(world, eid, Hitable);
        Hitable.resetHits(eid);
        Hitable.health[eid] = health;
    },
    hit$: obs((eid: number, secondEid: EntityId, forceMagnitude: number) => {
        const index = Hitable.hitIndex[eid] * 2;
        if (index === HITS_LIMIT) {
            console.warn(`[Hitable] Limit on hits`);
            return;
        }
        Hitable.hits.set(eid, index, secondEid);
        Hitable.hits.set(eid, index + 1, forceMagnitude);
        Hitable.hitIndex[eid] = index + 1;
    }),
    resetHits(eid: number) {
        Hitable.hitIndex[eid] = 0;
        Hitable.hits.getBatch(eid).fill(0);
    },
    getHitEids(eid: number) {
        return Hitable.hits.getBatch(eid)
            .subarray(0, Hitable.hitIndex[eid])
            .filter((_, i) => i % 2 === 0);
    },
    isDestroyed(eid: number) {
        return Hitable.health[eid] <= 0;
    },
}));
