import { NestedArray } from '../../utils.ts';
import { delegate } from '../../delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../utils.ts';

export const ROPE_BUFFER_LENGTH = 100;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

type RopeComponent = {
    points: ReturnType<typeof NestedArray.f64>;
    addComponent: (world: World, eid: number, points?: ArrayLike<number>) => void;
    set$: (eid: number, points: ArrayLike<number>) => void;
};

export const createRopeComponent = defineComponent<RopeComponent>(({ ref: Rope, obs }) => ({
        points: NestedArray.f64(ROPE_BUFFER_LENGTH, delegate.defaultSize),

        addComponent: (world: World, eid: number, points: ArrayLike<number> = []) => {
            addComponent(world, eid, Rope);
            Rope.points.getBatch(eid).fill(0);
            Rope.points.setBatch(eid, points);
        },
        set$: obs((eid: number, points: ArrayLike<number>) => {
            Rope.points.setBatch(eid, points);
        }),
}));
