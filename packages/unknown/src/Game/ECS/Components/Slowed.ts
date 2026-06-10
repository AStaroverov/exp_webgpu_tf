import { addComponent, EntityId, hasComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Presence = "this vehicle is slowed". `slowMul` ∈ [0, 1] is the freeze
 * amount: 0 = full speed (the dense default), 1 = fully frozen; the speed
 * sites scale movement impulse and turret turn speed by `1 - slowMul`.
 * Every Frost-kind damage event adds to it (cap 1); it thaws back each tick
 * in `createSlowedExpirySystem`, which removes the component at 0.
 */
export const createSlowedComponent = defineComponent((Slowed) => {
    const slowMul = TypedArray.f64(delegate.defaultSize);
    return {
        slowMul,
        addContribution(world: World, eid: EntityId, freeze: number) {
            if (!hasComponent(world, eid, Slowed)) {
                addComponent(world, eid, Slowed);
                slowMul[eid] = 0;
            }
            slowMul[eid] = Math.min(1, slowMul[eid] + freeze);
        },
    };
});
