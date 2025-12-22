import { EntityId } from "bitecs";
import { BehaviorSubject, map } from "rxjs";
import { Nil } from "../../../../../../../lib/Types";
import { dedobs, DEDOBS_REMOVE_DELAY, DEDOBS_RESET_DELAY } from "../../../../../../../lib/Rx/dedobs";

export const mapSlotToEid$ = new BehaviorSubject(new Map<number, Nil | EntityId>());
export const clearSlots = () => {
    mapSlotToEid$.next(new Map<number, Nil | EntityId>());
}

export const openSlot = (slot: number) => {
    mapSlotToEid$.next(mapSlotToEid$.value.set(slot, null));
}
export const closeSlot = (slot: number) => {
    mapSlotToEid$.value.delete(slot);
    mapSlotToEid$.next(mapSlotToEid$.value);
}
export const fillSlot = (slot: number, tankEid: EntityId) => {
    mapSlotToEid$.next(mapSlotToEid$.value.set(slot, tankEid));
}
export const getSlot = (slot: number) => {
    return mapSlotToEid$.value.get(slot);
}
export const getSlot$ =  dedobs(
    (slot: number) => {
        return mapSlotToEid$.pipe(
            map(() => getSlot(slot)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);

export const slotIsOpen$ = dedobs(
    (slot: number) => {
        return mapSlotToEid$.pipe(
            map((map) => map.has(slot)),
        );
    },
    {
        removeDelay: DEDOBS_REMOVE_DELAY,
        resetDelay: DEDOBS_RESET_DELAY,
    },
);
