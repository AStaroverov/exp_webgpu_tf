import { addEntity, EntityId } from 'bitecs';
import { GameDI } from '../../DI/GameDI';
import { TeamRef } from '../Components/TeamRef';

// Player is Person, is entitity that have control over a vehicle
export function createPlayer(teamId: number, { world } = GameDI): EntityId {
    const playerId = addEntity(world);

    TeamRef.addComponent(world, playerId, teamId);
    // Score.addComponent(world, playerId);

    return playerId;
}