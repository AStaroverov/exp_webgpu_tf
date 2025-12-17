import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

const MAX_CHILDREN = 1000;

export const Children = component({
    entitiesCount: new Float64Array(delegate.defaultSize),
    entitiesIds: NestedArray.f64(MAX_CHILDREN, delegate.defaultSize),

    addComponent(world: World, eid: number, ids: number[] | Float64Array = []): void {
        addComponent(world, eid, Children);
        Children.entitiesCount[eid] = ids.length;
        Children.entitiesIds.getBatch(eid).fill(0);
        Children.entitiesIds.setBatch(eid, ids);
    },

    addChildren(entity: number, child: number): void {
        const length = Children.entitiesCount[entity];

        if (length >= MAX_CHILDREN) {
            throw new Error('Max children reached');
        }

        Children.entitiesIds.set(entity, length, child);
        Children.entitiesCount[entity] += 1;
    },

    removeAllChildren(entity: number): void {
        Children.entitiesCount[entity] = 0;
        Children.entitiesIds.getBatch(entity).fill(0);
    },

    removeChild(parentEid: number, childEid: number): void {
        const children = Children.entitiesIds.getBatch(parentEid);
        const length = Children.entitiesCount[parentEid];

        if (length === 0) {
            return Children.removeAllChildren(parentEid);
        }

        const index = children.subarray(0, length).indexOf(childEid);

        if (index === -1) return;

        Children.entitiesCount[parentEid] -= 1;
        children.set(children.subarray(index + 1, length), index);
        children[Children.entitiesCount[parentEid]] = 0;
    },
});
