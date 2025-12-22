import { EntityId, innerQuery } from 'bitecs';
import { Vehicle, VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';
import { engine$, getEngine } from './engine.ts';
import { createPlayer } from '../../../Game/ECS/Entities/Player.ts';
import { map, shareReplay } from 'rxjs';
import { Pilot } from '../../../Pilots/Components/Pilot.ts';
import { TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';

export const GAME_MAX_TEAM_TANKS = 5;

export const playerId$ = engine$.pipe(
    map((e) => createPlayer(0, e)),
    shareReplay(1),
);

export function getVehicleEids() {
    return innerQuery(getEngine().world, [Vehicle]);
}

export function getVehicleType(vehicleEid: EntityId) {
    return Vehicle.type[vehicleEid] as VehicleType;
}

export function getPilotAgent(vehicleEid: EntityId) {
    return Pilot.getAgent(vehicleEid);
}

export function setPilotAgent(vehicleEid: EntityId, agent: TankAgent) {
    Pilot.addComponent(getEngine().world, vehicleEid, agent);
}
