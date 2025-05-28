import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';

export const PlayerRef = component({
    id: TypedArray.u32(delegate.defaultSize),

    addComponent: (world: World, entityId: EntityId, playerId: number) => {
        addComponent(world, entityId, PlayerRef);
        PlayerRef.id[entityId] = playerId;
    },
});
