import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { DamageKind } from './Damagable.ts';

/**
 * Rides a projectile: "stamp this damage-over-time on whatever part I hit".
 * The sibling of `Damagable` (instant damage) — together they fully describe
 * what a hit does. Becomes a `Dot` on the victim part.
 */
export const createDotableComponent = defineComponent((Dotable) => {
    const dps = TypedArray.f64(delegate.defaultSize);
    const kind = TypedArray.i8(delegate.defaultSize);
    const durationMs = TypedArray.f64(delegate.defaultSize);
    return {
        dps,
        kind,
        durationMs,
        addComponent(world: World, eid: EntityId, d: number, duration: number, dmgKind: DamageKind) {
            addComponent(world, eid, Dotable);
            dps[eid] = d;
            kind[eid] = dmgKind;
            durationMs[eid] = duration;
        },
    };
});
