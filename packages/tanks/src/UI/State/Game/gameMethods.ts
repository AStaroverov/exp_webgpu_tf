import { frameInterval } from '../../../../../../lib/Rx/frameInterval.ts';
import { BehaviorSubject, distinctUntilChanged, map } from 'rxjs';
import {
    getTankEngineLabel,
    getTankHealth,
    getTankHealthAbs,
    syncRemoveTank,
} from '../../../Game/ECS/Entities/Tank/TankUtils.ts';
import { addTank, getTankEids, getTankType } from './engineMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { EntityId } from 'bitecs';
import { getEngine } from './engine.ts';
import { hashArray } from '../../../../../../lib/hashArray.ts';
import { TankType } from '../../../Game/ECS/Components/Tank.ts';
import { PLAYER_TEAM_ID } from './playerMethods.ts';
import { CurrentActorAgent } from '../../../Pilots/Agents/CurrentActorAgent.ts';

export const mapSlotToEid$ = new BehaviorSubject(new Map<number, number>());

export function changeTankType(tankEid: EntityId, slot: number, tankType: TankType) {
    syncRemoveTank(tankEid);
    addTank(slot, PLAYER_TEAM_ID, tankType);
}

// export function changeTankPilot(tankEid: EntityId, slot: number, pilot: number) {
//     const tankEid = mapSlotToEid$.value.get(slot);
//
// }

export const tankEids$ = frameInterval(10).pipe(
    map(() => Array.from(getTankEids())),
    distinctUntilChanged((a, b) => {
        if (a.length !== b.length) return false;
        return hashArray(a) === hashArray(b);
    }),
);

export const getTankState$ = dedobs(
    (id: number) => {
        return frameInterval(200).pipe(
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

export const getTankType$ = dedobs(
    (id: number) => {
        return frameInterval(200).pipe(
            map(() => getTankType(id)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const finalizeGameState = async () => {
    const playerTeamEids = Array.from(getTankEids());

    for (let i = 0; i < playerTeamEids.length; i++) {
        addTank(i, 1, getTankType(playerTeamEids[i]));
    }

    for (const tankEid of getTankEids()) {
        const agent = new CurrentActorAgent(tankEid, false);
        getEngine().pilots.setPilot(tankEid, agent);
    }
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
