import { NestedArray } from '../../utils.ts';
import { delegate } from '../../delegate.ts';

export const ROPE_BUFFER_LENGTH = 100;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

export const Rope = ({
    points: NestedArray.f64(ROPE_BUFFER_LENGTH, delegate.defaultSize),
});

