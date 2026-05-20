import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createHeuristicsDataComponent = defineComponent((HeuristicsData) => {
    const approxColliderRadius = TypedArray.f64(delegate.defaultSize);
    return {
        approxColliderRadius,
        addComponent(world: World, eid: EntityId, radius: number) {
            addComponent(world, eid, HeuristicsData);
            approxColliderRadius[eid] = radius;
        },
    };
});
