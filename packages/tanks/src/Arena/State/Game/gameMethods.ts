import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { distinctUntilChanged, map } from 'rxjs';
import {
    getTankEngineLabel,
    getTankHealth,
    getTankHealthAbs,
    syncRemoveTank,
} from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { addTank, getPilotAgent, getVehicleEids, getVehicleType, setPilotAgent } from './engineMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { EntityId } from 'bitecs';
import { getEngine } from './engine.ts';
import { hashArray } from '../../../../../../lib/hashArray.ts';
import { TankVehicleType } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { CurrentActorAgent, TankAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';
import { Nil } from '../../../../../../lib/Types/index.ts';
import { getPilotAgents } from '../../../Pilots/Components/Pilot.ts';
import { PLAYER_TEAM_ID } from './def.ts';
import { fillSlot, mapSlotToEid$ } from './modules/lobbySlots.ts';

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

export const finalizeGameState = async () => {
    const playerTeamEids = Array.from(mapSlotToEid$.value.values())
        .filter((eid): eid is EntityId => eid != null);

    for (let i = 0; i < playerTeamEids.length; i++) {
        const vehicleEid = addTank(i, 1, getVehicleType(playerTeamEids[i]) as TankVehicleType);
        const agent = new CurrentActorAgent(vehicleEid, false);
        setPilotAgent(vehicleEid, agent);
    }

    // Sync all AI agents to load TensorFlow models
    const pilots = getPilotAgents();
    await Promise.all(pilots.map(pilot => pilot?.sync ? pilot.sync() : Promise.resolve())); 
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
