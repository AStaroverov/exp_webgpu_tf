import { EntityId, innerQuery } from 'bitecs';
import { Vehicle, VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';
import { Tank } from '../../../Game/ECS/Components/Tank.ts';
import { TeamRef } from '../../../Game/ECS/Components/TeamRef.ts';
import { PlayerRef } from '../../../Game/ECS/Components/PlayerRef.ts';
import { Score } from '../../../Game/ECS/Components/Score.ts';
import { Color } from '../../../../../renderer/src/ECS/Components/Common.ts';
import { getTankHealth } from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { engine$, getEngine } from './engine.ts';
import { createPlayer } from '../../../Game/ECS/Entities/Player.ts';
import { map, shareReplay } from 'rxjs';
import { Pilot } from '../../../Pilots/Components/Pilot.ts';
import { TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';

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

export function getTankEids(): EntityId[] {
    return Array.from(innerQuery(getEngine().world, [Vehicle, Tank]));
}

export function getTankInfo(eid: EntityId): TankInfo {
    return {
        eid,
        health: getTankHealth(eid),
        teamId: TeamRef.id[eid],
        playerId: PlayerRef.id[eid],
        color: [Color.getR(eid), Color.getG(eid), Color.getB(eid)],
    };
}

export function getPlayerScore(playerId: EntityId): { spice: number; debris: number } {
    return {
        spice: Score.spices[playerId] ?? 0,
        debris: Score.debris[playerId] ?? 0,
    };
}
