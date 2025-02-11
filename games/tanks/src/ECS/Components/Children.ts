import { defineComponent, Types } from 'bitecs';

export const Children = defineComponent({
    entitiesCount: Types.f64,
    entitiesIds: [Types.f64, 1000],
});