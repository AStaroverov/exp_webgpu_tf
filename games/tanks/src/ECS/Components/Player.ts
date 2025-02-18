import { addComponent, defineComponent, Types } from 'bitecs';
import { DI } from '../../DI';

export const Player = defineComponent({
    id: Types.f64,
});

let playerId = 0;

export function getNewPlayerId() {
    return playerId++;
}

export function addPlayerComponent(entityId: number, playerId: number, { world } = DI) {
    addComponent(world, Player, entityId);
    Player.id[playerId] = playerId;
}