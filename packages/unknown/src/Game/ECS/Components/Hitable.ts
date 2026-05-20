import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

const HITS_LIMIT = 10;

export const createHitableComponent = defineComponent((Hitable, obs) => {
    const health = TypedArray.f64(delegate.defaultSize);
    const hitIndex = TypedArray.i8(delegate.defaultSize);
    const hits = NestedArray.f64(2 * HITS_LIMIT, delegate.defaultSize);

    function resetHits(eid: number) {
        hitIndex[eid] = 0;
        hits.getBatch(eid).fill(0);
    }

    return {
        health,
        hitIndex,
        hits,

        addComponent(world: World, eid: number, hp: number) {
            addComponent(world, eid, Hitable);
            resetHits(eid);
            health[eid] = hp;
        },
        hit$: obs((eid: number, secondEid: EntityId, forceMagnitude: number) => {
            const index = hitIndex[eid] * 2;
            if (index === HITS_LIMIT) {
                console.warn(`[Hitable] Limit on hits`);
                return;
            }
            hits.set(eid, index, secondEid);
            hits.set(eid, index + 1, forceMagnitude);
            hitIndex[eid] = index + 1;
        }),
        resetHits,
        getHitEids(eid: number): Float64Array {
            return hits.getBatch(eid)
                .subarray(0, hitIndex[eid])
                .filter((_: number, i: number) => i % 2 === 0) as Float64Array;
        },
        isDestroyed(eid: number): boolean {
            return health[eid] <= 0;
        },
    };
});
