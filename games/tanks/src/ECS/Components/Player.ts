import { addComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { delegate } from '../../../../../src/delegate.ts';

export const Player = ({
    id: new Float64Array(delegate.defaultSize),
});

let playerId = 0;

export function getNewPlayerId() {
    return playerId++;
}

export function addPlayerComponent(entityId: number, playerId: number, { world } = GameDI) {
    addComponent(world, entityId, Player);
    Player.id[entityId] = playerId;
}