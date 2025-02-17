import { defineComponent, Types } from 'bitecs';

const MAX_CHILDREN = 1000;

export const Children = defineComponent({
    entitiesCount: Types.f64,
    entitiesIds: [Types.f64, MAX_CHILDREN],
});

export function addChildren(entity: number, child: number) {
    const length = Children.entitiesCount[entity];

    if (length >= MAX_CHILDREN) {
        throw new Error('Max children reached');
    }

    Children.entitiesIds[entity][length] = child;
    Children.entitiesCount[entity] += 1;
}

export function removeAllChildren(entity: number) {
    Children.entitiesCount[entity] = 0;
}

export function removeChild(entity: number, child: number) {
    const children = Children.entitiesIds[entity];
    const length = Children.entitiesCount[entity];
    const index = children.subarray(0, length).indexOf(child);

    if (index === -1) return;

    Children.entitiesCount[entity] -= 1;
    children.set(children.subarray(0, index), 0);
    children.set(children.subarray(index + 1, length), index);
    children[Children.entitiesCount[entity]] = 0;
}