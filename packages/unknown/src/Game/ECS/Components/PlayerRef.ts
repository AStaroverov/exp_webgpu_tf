import { addComponent, EntityId, removeComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createPlayerRefComponent = defineComponent((PlayerRef) => {
    const id = TypedArray.u32(delegate.defaultSize);
    return {
        id,
        addComponent(world: World, eid: EntityId, playerId: number) {
            addComponent(world, eid, PlayerRef);
            id[eid] = playerId;
        },
        removeComponent(world: World, eid: number) {
            id[eid] = 0;
            removeComponent(world, eid, PlayerRef);
        },
    };
});
