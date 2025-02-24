import { component, NestedArray, obs } from '../../utils.ts';
import { delegate } from '../../delegate.ts';
import { addComponent, World } from 'bitecs';

export const ROPE_BUFFER_LENGTH = 100;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

export const Rope = component({
    points: NestedArray.f64(ROPE_BUFFER_LENGTH, delegate.defaultSize),

    addComponent: (world: World, eid: number, points: ArrayLike<number> = []) => {
        addComponent(world, eid, Rope);
        Rope.points.setBatch(eid, points);
    },
    set$: obs((eid: number, points: ArrayLike<number>) => {
        Rope.points.setBatch(eid, points);
    }),
});
