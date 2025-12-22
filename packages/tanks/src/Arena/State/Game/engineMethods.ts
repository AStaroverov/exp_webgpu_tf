import { GameDI } from '../../../Game/DI/GameDI.ts';
import { PI } from '../../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { EntityId, innerQuery } from 'bitecs';
import { Vehicle, VehicleType } from '../../../Game/ECS/Components/Vehicle.ts';
import { engine$, getEngine } from './engine.ts';
import { createTank, TankVehicleType } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { createHarvester } from '../../../Game/ECS/Entities/Harvester/Harvester.ts';
import { createPlayer } from '../../../Game/ECS/Entities/Player.ts';
import { map, shareReplay } from 'rxjs';
import { getValue } from '../../../../../../lib/Rx/getValue.ts';
import { Pilot } from '../../../Pilots/Components/Pilot.ts';
import { TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { PLAYER_TEAM_ID } from './def.ts';

export const GAME_MAX_TEAM_TANKS = 5;

export const playerId$ = engine$.pipe(
    map((e) => createPlayer(0, e)),
    shareReplay(1),
);

function getSlotPosition(slot: number, teamId: number) {
    const x = GameDI.width * 0.2 + (teamId === 1 ? GameDI.width * 0.6 : 0);
    const dy = (GameDI.height - GameDI.height * 0.4) / (GAME_MAX_TEAM_TANKS - 1);
    const y = GameDI.height * 0.2 + slot * dy;
    return { x, y };
}

export function addTank(slot: number, teamId: number, vehicleType: TankVehicleType) {
    const { x, y } = getSlotPosition(slot, teamId);

    return createTank({
        type: vehicleType,
        playerId: getValue(playerId$),
        teamId,
        x,
        y,
        rotation: PI / 2 + randomRangeFloat(-PI / 4, PI / 4),
        color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });
}

export function addHarvester() {
    return createHarvester({
        playerId: getValue(playerId$),
        teamId: PLAYER_TEAM_ID,
        x: 600,
        y: 600,
        rotation: PI / 2 + randomRangeFloat(-PI / 4, PI / 4),
        color: [PLAYER_TEAM_ID, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });
}

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
