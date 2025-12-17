import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { BehaviorSubject, distinctUntilChanged, map } from 'rxjs';
import {
    getTankEngineLabel,
    getTankHealth,
    getTankHealthAbs,
    syncRemoveTank,
} from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { addTank, getVehicleEids, getVehicleType } from './engineMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { EntityId } from 'bitecs';
import { getEngine } from './engine.ts';
import { hashArray } from '../../../../../../lib/hashArray.ts';
import { TankVehicleType } from '../../../Game/ECS/Entities/Tank/createTank.ts';
import { PLAYER_TEAM_ID } from './playerMethods.ts';
import { CurrentActorAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';

export const mapSlotToEid$ = new BehaviorSubject(new Map<number, number>());

export function changeTankType(vehicleEid: EntityId, slot: number, vehicleType: TankVehicleType) {
    syncRemoveTank(vehicleEid);
    addTank(slot, PLAYER_TEAM_ID, vehicleType);
}

// export function changeTankPilot(tankEid: EntityId, slot: number, pilot: number) {
//     const tankEid = mapSlotToEid$.value.get(slot);
//
// }

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

export const finalizeGameState = async () => {
    const playerTeamEids = Array.from(getVehicleEids());

    for (let i = 0; i < playerTeamEids.length; i++) {
        addTank(i, 1, getVehicleType(playerTeamEids[i]) as TankVehicleType);
    }

    for (const vehicleEid of getVehicleEids()) {
        const agent = new CurrentActorAgent(vehicleEid, false);
        getEngine().pilots.setPilot(vehicleEid, agent);
    }

    // Sync all AI agents to load TensorFlow models
    const pilots = getEngine().pilots.getPilots();
    await Promise.all(
        pilots.map(pilot => pilot.sync ? pilot.sync() : Promise.resolve())
    );
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
