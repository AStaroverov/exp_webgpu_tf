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
import { removeTankPilot } from './pilotsMethods.ts';
import { PilotType } from '../../../Pilots/Components/Pilot.ts';

export const mapSlotToEid$ = new BehaviorSubject(new Map<number, number>());

export function changeTankType(tankEid: EntityId, slot: number, tankType: TankType) {
    syncRemoveTank(tankEid);
    removeTankPilot(tankEid);
    addTank(slot, PLAYER_TEAM_ID, tankType);
}

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

export const finalizeGameState = () => {
    const playerTeamEids = Array.from(getTankEids());

    for (let i = 0; i < playerTeamEids.length; i++) {
        const tankEid = addTank(i, 1, getTankType(playerTeamEids[i]));
        getEngine().pilots.setPilot(tankEid, PilotType.Agent);
    }
};

export const resetGameState = () => {
    mapSlotToEid$.value.clear();
    mapSlotToEid$.next(mapSlotToEid$.value);
};

export function activateBots() {
    getEngine().pilots.toggle(true);
}

export function deactivateBots() {
    getEngine().pilots.toggle(false);
}
