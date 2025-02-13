import { addComponent, defineComponent, IWorld, Types } from 'bitecs';

export const Player = defineComponent({
    id: Types.f64,
});

let playerId = 0;

export function getNewPlayerId() {
    return playerId++;
}

export function addPlayerComponent(world: IWorld, entityId: number, playerId: number) {
    addComponent(world, Player, entityId);
    Player.id[playerId] = playerId;
}