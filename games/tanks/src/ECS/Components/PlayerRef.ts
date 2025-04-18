import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';
import { TypedArray } from '../../../../../src/utils.ts';

export const PlayerRef = component({
    id: TypedArray.u32(delegate.defaultSize),

    addComponent: (world: World, entityId: EntityId, playerId: number) => {
        addComponent(world, entityId, PlayerRef);
        PlayerRef.id[entityId] = playerId;
    },
});
