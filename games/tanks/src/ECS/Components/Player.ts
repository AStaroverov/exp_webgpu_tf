import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';

export const Player = component({
    id: new Float64Array(delegate.defaultSize),

    addComponent: (world: World, entityId: EntityId, playerId: number) => {
        addComponent(world, entityId, Player);
        Player.id[entityId] = playerId;
    },
});

let playerId = 0;

export function getNewPlayerId() {
    return playerId++;
}
