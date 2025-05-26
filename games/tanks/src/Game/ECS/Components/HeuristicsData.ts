import { component } from '../../../../../../src/ECS/utils.ts';
import { TypedArray } from '../../../../../../src/utils.ts';
import { delegate } from '../../../../../../src/delegate.ts';
import { addComponent, EntityId, World } from 'bitecs';

export const MAX_APPROXIMATE_COLLIDER_RADIUS = 500;

export const HeuristicsData = component({
    // it's a aproximation of the collision radius, use for simple heuristics
    approxColliderRadius: TypedArray.f64(delegate.defaultSize),

    addComponent: (world: World, eid: EntityId, radius: number) => {
        addComponent(world, eid, HeuristicsData);
        HeuristicsData.approxColliderRadius[eid] = radius;
    },
});
