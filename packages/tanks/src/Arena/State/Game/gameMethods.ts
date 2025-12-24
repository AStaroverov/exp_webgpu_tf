import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { distinctUntilChanged, map } from 'rxjs';
import {
    getTankEngineLabel,
    getTankHealth,
    getTankHealthAbs,
    syncRemoveTank,
} from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { getPilotAgent, getPlayerScore, getTankEids, getTankInfo, getVehicleEids, getVehicleType, playerId$, setPilotAgent } from './engineMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { EntityId } from 'bitecs';
import { getEngine } from './engine.ts';
import { hashArray } from '../../../../../../lib/hashArray.ts';
import { createTank, TankVehicleType } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { CurrentActorAgent, TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { Nil } from '../../../../../../lib/Types/index.ts';
import { getRegistratedAgents } from '../../../Pilots/Components/Pilot.ts';
import { PLAYER_TEAM_ID } from './def.ts';
import { fillSlot, mapSlotToEid$ } from './modules/lobbySlots.ts';
import { PI } from '../../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createHarvester } from '../../../Game/ECS/Entities/Harvester/Harvester.ts';
import { createRock } from '../../../Game/ECS/Entities/Rock/Rock.ts';
import { getTeamSpawnPosition, allocateSpawnCell, CellContent, findNextAvailableSlot, getSpawnGrid, isCellEmpty, getCellWorldPosition, setCellContent } from './SpawnGrid.ts';
import { getValue } from '../../../../../../lib/Rx/getValue.ts';
import { spawnSpiceCluster } from '../../../Game/ECS/Entities/Spice/Spice.ts';
import { createSpiceCollector } from '../../../Game/ECS/Entities/SpiceCollector/SpiceCollector.ts';

export function addTank(slot: number, teamId: number, vehicleType: TankVehicleType) {
    const { x, y } = getTeamSpawnPosition(teamId, slot + 1);

    const entity = createTank({
        type: vehicleType,
        playerId: getValue(playerId$),
        teamId,
        x,
        y,
        rotation: PI / 2 + randomRangeFloat(-PI / 4, PI / 4),
        color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    allocateSpawnCell(teamId, slot, CellContent.Vehicle, entity);

    return entity;
}

export function addHarvester(teamId: number = PLAYER_TEAM_ID) {
    const slot = findNextAvailableSlot(teamId);
    if (slot === -1) {
        throw new Error('No available spawn slots for harvester');
    }

    const { x, y } = getTeamSpawnPosition(teamId, slot);

    const entity = createHarvester({
        playerId: getValue(playerId$),
        teamId,
        x,
        y,
        rotation: PI / 2 + randomRangeFloat(-PI / 4, PI / 4),
        color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    createSpiceCollector({
        x,
        y,
        teamId,
    });

    allocateSpawnCell(teamId, slot, CellContent.Vehicle, entity);

    return entity;
}

export function addFauna() {
    const spawnProbability = {
        rock: 0.3,
        spice: 0.1,
    }

    const grid = getSpawnGrid();

    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            if (!isCellEmpty(col, row)) continue;
            if (Math.random() < spawnProbability.rock) {
                const { x, y } = getCellWorldPosition(col, row);
                createRock({ x, y });
                setCellContent(col, row, CellContent.Obstacle);
            } else if (Math.random() < spawnProbability.spice) {
                const { x, y } = getCellWorldPosition(col, row);
                spawnSpiceCluster({ x, y });
                setCellContent(col, row, CellContent.Obstacle);
            }
        }
    }
}

export function changeTankType(vehicleEid: Nil | EntityId, slot: number, vehicleType: TankVehicleType) {
    const agent = vehicleEid ? getPilotAgent(vehicleEid) : undefined;

    if (vehicleEid) syncRemoveTank(vehicleEid);
    const tankEid = addTank(slot, PLAYER_TEAM_ID, vehicleType);
    fillSlot(slot, tankEid);

    if (agent) setPilotAgent(tankEid, agent);
}

export function changeTankPilot(vehicleEid: EntityId, agent: TankAgent) {
    setPilotAgent(vehicleEid, agent);
}

export const vehicleEids$ = frameInterval(160).pipe(
    map(() => Array.from(getVehicleEids())),
    distinctUntilChanged((a, b) => {
        if (a.length !== b.length) return false;
        return hashArray(a) === hashArray(b);
    }),
);

export const getVehicleState$ = dedobs(
    (id: number) => {
        return frameInterval(32 * 100).pipe(
            map(() => {
                return {
                    id,
                    healthRel: getTankHealth(id),
                    healthAbs: (getTankHealthAbs(id)).toFixed(0),
                    engine: getTankEngineLabel(id),
                };
            }),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const getVehicleType$ = dedobs(
    (id: number) => {
        return frameInterval(32 * 100).pipe(
            map(() => getVehicleType(id)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const getPilotAgent$ = dedobs(
    (id: number) => {
        return frameInterval(32 * 100).pipe(
            map(() => getPilotAgent(id)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const tankEids$ = frameInterval(160).pipe(
    map(() => getTankEids()),
    distinctUntilChanged((a, b) => {
        if (a.length !== b.length) return false;
        return hashArray(a) === hashArray(b);
    }),
);

export const getTankInfo$ = dedobs(
    (eid: number) => {
        return frameInterval(100).pipe(
            map(() => getTankInfo(eid)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const playerScore$ = frameInterval(100).pipe(
    map(() => getPlayerScore(getValue(playerId$))),
);

export const finalizeGameState = async () => {
    const playerTeamEids = Array.from(mapSlotToEid$.value.values())
        .filter((eid): eid is EntityId => eid != null);

    for (let i = 0; i < playerTeamEids.length; i++) {
        const vehicleEid = addTank(i, 1, getVehicleType(playerTeamEids[i]) as TankVehicleType);
        const agent = new CurrentActorAgent(vehicleEid, false);
        setPilotAgent(vehicleEid, agent);
    }

    // Sync all AI agents to load TensorFlow models
    const agents = getRegistratedAgents();
    await Promise.all(agents.map(agent => agent?.sync ? agent.sync() : Promise.resolve())); 
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
