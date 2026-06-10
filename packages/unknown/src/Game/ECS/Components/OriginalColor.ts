import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/**
 * Existence-based save-slot for a tinted part: snapshot of the part's
 * pre-tint live color. Presence = "this part is recolored"; the tint system
 * restores from here and removes the component when no status remains.
 */
export const createOriginalColorComponent = defineComponent((OriginalColor) => {
    const r = TypedArray.f64(delegate.defaultSize);
    const g = TypedArray.f64(delegate.defaultSize);
    const b = TypedArray.f64(delegate.defaultSize);
    return {
        r,
        g,
        b,
        addComponent(world: World, eid: EntityId, cr: number, cg: number, cb: number) {
            addComponent(world, eid, OriginalColor);
            r[eid] = cr;
            g[eid] = cg;
            b[eid] = cb;
        },
    };
});
