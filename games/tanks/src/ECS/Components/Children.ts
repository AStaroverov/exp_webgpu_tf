import { addComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { NestedArray } from '../../../../../src/utils.ts';

const MAX_CHILDREN = 1000;

export const Children = ({
    entitiesCount: new Float64Array(delegate.defaultSize),
    entitiesIds: NestedArray.f64(MAX_CHILDREN, delegate.defaultSize),
});

export function addChildrenComponent(entity: number, count: number = 0, ids: number[] | Float64Array = [], { world } = GameDI) {
    addComponent(world, entity, Children);
    Children.entitiesCount[entity] = count;
    Children.entitiesIds.setBatch(entity, ids);
}

export function addChildren(entity: number, child: number) {
    const length = Children.entitiesCount[entity];

    if (length >= MAX_CHILDREN) {
        throw new Error('Max children reached');
    }

    Children.entitiesIds.set(entity, length, child);
    Children.entitiesCount[entity] += 1;
}

export function removeAllChildren(entity: number) {
    Children.entitiesCount[entity] = 0;
}

export function removeChild(entity: number, child: number) {
    const children = Children.entitiesIds.getBatch(entity);
    const length = Children.entitiesCount[entity];
    const index = children.subarray(0, length).indexOf(child);

    if (index === -1) return;

    Children.entitiesCount[entity] -= 1;
    children.set(children.subarray(0, index), 0);
    children.set(children.subarray(index + 1, length), index);
    children[Children.entitiesCount[entity]] = 0;
}