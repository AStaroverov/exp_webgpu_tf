import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

/** Max hexes a single obstacle may occupy. */
const FOOTPRINT_LIMIT = 16;

/**
 * The set of hex cells (axial q,r) a static obstacle occupies, stored on the
 * obstacle's parent entity. Mirrors the `Children` pattern (count + flat pairs).
 *
 * Kept so a future destroy system can `vacate()` exactly the cells this obstacle
 * claimed (see OBSTACLES_HEX_INTEGRATION_PLAN.md, Decision 5 — deferred).
 */
export const createObstacleFootprintComponent = defineComponent((ObstacleFootprint) => {
    const count = TypedArray.u8(delegate.defaultSize);
    // Flat (q, r) pairs per entity.
    const cells = NestedArray.f64(2 * FOOTPRINT_LIMIT, delegate.defaultSize);

    return {
        count,
        cells,

        addComponent(world: World, eid: number) {
            addComponent(world, eid, ObstacleFootprint);
            count[eid] = 0;
        },

        add(eid: number, q: number, r: number) {
            const len = count[eid];
            if (len >= FOOTPRINT_LIMIT) {
                throw new Error('ObstacleFootprint limit reached');
            }
            cells.set(eid, len * 2, q);
            cells.set(eid, len * 2 + 1, r);
            count[eid] = len + 1;
        },

        forEach(eid: number, fn: (q: number, r: number) => void) {
            const len = count[eid];
            for (let i = 0; i < len; i++) {
                fn(cells.get(eid, i * 2), cells.get(eid, i * 2 + 1));
            }
        },
    };
});
