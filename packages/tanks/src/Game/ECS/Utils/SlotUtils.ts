import { EntityId, hasComponent } from 'bitecs';
import { Children } from '../Components/Children.ts';
import { Slot } from '../Components/Slot.ts';
import { GameDI } from '../../DI/GameDI.ts';

export function isSlot(eid: EntityId, { world } = GameDI): boolean {
    const hasSlot = hasComponent(world, eid, Slot);
    const hasChildren = hasComponent(world, eid, Children);
    if (hasSlot && !hasChildren) {
        throw new Error(`Children component not found for slot entity ${eid}`);
    }
    return hasSlot && hasChildren;
}

export function isSlotFilled(slotEid: EntityId): boolean {
    return Children.entitiesCount[slotEid] > 0;
}

export function isSlotEmpty(slotEid: EntityId): boolean {
    return Children.entitiesCount[slotEid] === 0;
}

export function getSlotFillerEid(slotEid: EntityId): EntityId {
    return Children.entitiesIds.get(slotEid, 0);
}
