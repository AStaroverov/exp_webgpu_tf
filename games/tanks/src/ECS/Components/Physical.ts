import { defineComponent, Types } from 'bitecs';

export const RigidBodyRef = defineComponent({
    id: Types.f64,
    x: Types.f64,
    y: Types.f64,
    rotation: Types.f64,
});
