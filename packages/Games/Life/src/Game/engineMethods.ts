import { EntityId } from 'bitecs';
import { engine$, getEngine } from './engine.js';
import { map, shareReplay } from 'rxjs';
import { TankAgent } from '../Plugins/Pilots/Agents/CurrentActorAgent.js';
import { Pilot } from '../Plugins/Pilots/Components/Pilot.js';
import { createPlayer } from '../GameEngine/ECS/Entities/Player.js';

export const GAME_MAX_TEAM_TANKS = 5;

export type TankInfo = {
    eid: number;
    health: number;
    teamId: number;
    playerId: number;
    color: [number, number, number];
};

export type TeamScoreInfo = {
    teamId: number;
    spice: number;
    debris: number;
};

export const playerId$ = engine$.pipe(
    map((e) => createPlayer(0, e)),
    shareReplay(1),
);

export function getPilotAgent(vehicleEid: EntityId) {
    return Pilot.getAgent(vehicleEid);
}

export function setPilotAgent(vehicleEid: EntityId, agent: TankAgent) {
    Pilot.addComponent(getEngine().world, vehicleEid, agent);
}
