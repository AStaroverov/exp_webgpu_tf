import { getEngine } from './engine.ts';
import { ValueOf } from '../../../../../../lib/Types';
import { mapSlotToEid$ } from './gameMethods.ts';
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from '../../../../../../lib/Rx/dedobs.ts';
import { interval, map, mergeMap, of } from 'rxjs';
import { getPilotType, PilotType } from '../../../Pilots/Components/Pilot.ts';
import { getTankEidBySlot$ } from './playerMethods.ts';

export const tankPilotType$ = dedobs((slot: number) => {
    return mapSlotToEid$.pipe(
        mergeMap((slots) => {
            return slots.has(slot)
                ? interval(100).pipe(
                    map(() => getEngine().pilots.getPilotType(slots.get(slot)!)),
                )
                : of(null);
        }),
    );
}, {
    removeDelay: DEDOBS_REMOVE_DELAY,
    resetDelay: DEDOBS_RESET_DELAY,
});

export function changeTankPilotBySlot(slot: number, version: ValueOf<typeof PilotType>) {
    const tankEid = mapSlotToEid$.value.get(slot);

    if (tankEid == null) {
        console.warn(`Tank with slot ${ slot } not found`);
        return;
    }

    if (version === 0) {
        getEngine().pilots.setPlayerPilot(tankEid);
        return;
    }

    getEngine().pilots.setPilot(tankEid, version);
}

export const getPilotTypeBySlot$ = dedobs((slot: number) => {
    return getTankEidBySlot$(slot).pipe(
        mergeMap((eid) => eid == null
            ? of(undefined)
            : interval(100).pipe(map(() => {
                return getPilotType(eid);
            }))),
    );
}, {
    removeDelay: DEDOBS_REMOVE_DELAY,
    resetDelay: DEDOBS_RESET_DELAY,
});

export function removeTankPilot(tankEid: number) {
    getEngine().pilots.removePilot(tankEid);
}
