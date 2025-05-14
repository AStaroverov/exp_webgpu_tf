import { addEntity, EntityId } from 'bitecs';
import { Score } from '../Components/Score.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { TeamRef } from '../Components/TeamRef.ts';

export function createPlayer(teamId: number, { world } = GameDI): EntityId {
    const playerId = addEntity(world);

    TeamRef.addComponent(world, playerId, teamId);
    Score.addComponent(world, playerId);

    return playerId;
}