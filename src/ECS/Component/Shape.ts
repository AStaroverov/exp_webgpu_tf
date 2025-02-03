import { defineComponent, Types } from 'bitecs';

export const Shape = defineComponent({
    kind: Types.ui32, // 0: circle, 1: segment, 2: rectangle
});