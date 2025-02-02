import { defineComponent, Types } from 'bitecs';

export const ROPE_BUFFER_LENGTH = 20;
export const ROPE_POINTS_COUNT = ROPE_BUFFER_LENGTH / 2;
export const ROPE_SEGMENTS_COUNT = ROPE_POINTS_COUNT - 1;

export const Rope = defineComponent({
    points: [Types.f32, ROPE_BUFFER_LENGTH],
});