import { NestedArray } from '../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../renderer/src/ECS/utils.ts';
import { MapWorldId } from '../../../Map/HexGrid.ts';
import { TargetKind } from '../ActionTypes.ts';

/**
 * ActionTarget — addressing ("what we act on"). A `kind` discriminator plus a
 * generic `values` bag (interpretation depends on `kind`), mirroring the Shape
 * component: adding a target kind is one enum entry + one setter, no new fields.
 *
 * values layout per kind:
 *   Entity → [eid, worldId]
 *   Hex    → [q,   r]
 *   Point  → [x,   y]
 */
export const createActionTargetComponent = defineComponent((ActionTarget, obs) => {
    const kind = new Uint8Array(delegate.defaultSize);
    const values = NestedArray.f64(2, delegate.defaultSize);
    return {
        kind,
        values,
        addComponent(world: World, eid: number) {
            addComponent(world, eid, ActionTarget);
            kind[eid] = TargetKind.None;
            values.set(eid, 0, 0);
            values.set(eid, 1, 0);
        },
        setEntity$: obs((eid: number, target: number, worldId: MapWorldId = MapWorldId.Game) => {
            kind[eid] = TargetKind.Entity;
            values.set(eid, 0, target);
            values.set(eid, 1, worldId);
        }),
        setHex$: obs((eid: number, q: number, r: number) => {
            kind[eid] = TargetKind.Hex;
            values.set(eid, 0, q);
            values.set(eid, 1, r);
        }),
        setPoint$: obs((eid: number, x: number, y: number) => {
            kind[eid] = TargetKind.Point;
            values.set(eid, 0, x);
            values.set(eid, 1, y);
        }),
    };
});
