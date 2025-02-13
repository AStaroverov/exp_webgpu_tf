import { defineComponent, Types } from 'bitecs';

const MAX_CHILDREN = 1000;

export const Children = defineComponent({
    entitiesCount: Types.f64,
    entitiesIds: [Types.f64, MAX_CHILDREN],
});

export function removeAllChildren(entity: number) {
    Children.entitiesCount[entity] = 0;
}

export function removeChild(entity: number, child: number) {
    const children = Children.entitiesIds[entity];
    const index = children.indexOf(child);

    if (index === -1) return;

    Children.entitiesCount[entity] -= 1;
    Children.entitiesIds[entity].set(Children.entitiesIds[entity].subarray(0, index), 0);
    Children.entitiesIds[entity].set(Children.entitiesIds[entity].subarray(index + 1, MAX_CHILDREN), index);
    Children.entitiesIds[entity][Children.entitiesCount[entity]] = 0;
}