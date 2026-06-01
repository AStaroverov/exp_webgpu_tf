import { EntityId, hasComponent } from 'bitecs';
import { getSlotWorldComponents, SlotWorld } from '../createSlotWorld.ts';
import { getOccupantOf } from '../refs.ts';

export function isSlot(world: SlotWorld, eid: EntityId): boolean {
    const { Slot } = getSlotWorldComponents(world);
    return hasComponent(world, eid, Slot);
}

export function isSlotFilled(_world: SlotWorld, slotEid: EntityId): boolean {
    return getOccupantOf(slotEid) !== 0;
}

export function isSlotEmpty(_world: SlotWorld, slotEid: EntityId): boolean {
    return getOccupantOf(slotEid) === 0;
}

// Returns the PART PHYS atom eid occupying the slot (0 = empty).
export function getSlotFillerEid(_world: SlotWorld, slotEid: EntityId): EntityId {
    return getOccupantOf(slotEid);
}
