import { createMethods, NestedArray } from '../../utils.ts';
import { delegate } from '../../delegate.ts';
import { addComponent } from 'bitecs';

export const ROPE_BUFFER_LENGTH = 100;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

export const Rope = ({
    points: NestedArray.f64(ROPE_BUFFER_LENGTH, delegate.defaultSize),
});

export const RopeMethods = createMethods(Rope, {
    addComponent(world, eid, comp, points: ArrayLike<number> = []) {
        addComponent(world, eid, comp);
        comp.points[eid].fill(points);
    },
    set$(comp, eid: number, points: ArrayLike<number>) {
        comp.points[eid].fill(points);
    },
});

