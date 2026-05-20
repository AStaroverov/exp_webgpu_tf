import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { addComponent, EntityId, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export enum TrackSide {
    Left = 0,
    Right = 1,
}

export const createTrackComponent = defineComponent((Track) => {
    const side = TypedArray.i8(delegate.defaultSize);
    const length = TypedArray.f64(delegate.defaultSize);
    return {
        side,
        length,
        addComponent(world: World, eid: EntityId, s: TrackSide, len: number) {
            addComponent(world, eid, Track);
            length[eid] = len;
            side[eid] = s;
        },
    };
});
