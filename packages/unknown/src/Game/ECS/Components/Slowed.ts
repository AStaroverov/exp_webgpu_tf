import { addComponent, EntityId, hasComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Presence = "this vehicle is slowed". `slowMul` ∈ (0, 1) scales both the
 * movement impulse and the turret turn speed; it is the AVERAGE of all
 * contributions (one per Frost-kind damage event, added by the hitable
 * pipeline). Every contribution refreshes `remaining`; expired by
 * `createSlowedExpirySystem`.
 */
export const createSlowedComponent = defineComponent((Slowed) => {
    const slowMul = TypedArray.f64(delegate.defaultSize);
    const slowSum = TypedArray.f64(delegate.defaultSize);
    const count = TypedArray.f64(delegate.defaultSize);
    const remaining = TypedArray.f64(delegate.defaultSize);
    return {
        slowMul,
        remaining,
        addContribution(world: World, eid: EntityId, slow: number, durationMs: number) {
            if (!hasComponent(world, eid, Slowed)) {
                addComponent(world, eid, Slowed);
                slowSum[eid] = 0;
                count[eid] = 0;
            }
            slowSum[eid] += slow;
            count[eid] += 1;
            slowMul[eid] = slowSum[eid] / count[eid];
            remaining[eid] = durationMs;
        },
    };
});
