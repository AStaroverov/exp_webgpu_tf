import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';

export enum TrackSide {
    Left = 0,
    Right = 1,
}

/**
 * Component for a track/caterpillar entity.
 * Each track is an independent drive unit attached to the vehicle via joint.
 */
export const Track = component({
    // Track side (left or right)
    side: TypedArray.i8(delegate.defaultSize),
    // Track length for animation calculations
    length: TypedArray.f64(delegate.defaultSize),

    addComponent(world: World, eid: EntityId, side: TrackSide, length: number): void {
        addComponent(world, eid, Track);
        Track.length[eid] = length;
        Track.side[eid] = side;
    }
});

