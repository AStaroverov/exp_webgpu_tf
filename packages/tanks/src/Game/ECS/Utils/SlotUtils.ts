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
    if (!isSlot(slotEid)) {
        throw new Error(`Slot component not found for entity ${slotEid}`);
    }
    return Children.entitiesCount[slotEid] > 0;
}

export function isSlotEmpty(slotEid: EntityId): boolean {
    if (!isSlot(slotEid)) {
        throw new Error(`Slot component not found for entity ${slotEid}`);
    }
    return Children.entitiesCount[slotEid] === 0;
}

export function getSlotFillerEid(slotEid: EntityId): EntityId {
    if (!isSlot(slotEid)) {
        throw new Error(`Slot component not found for entity ${slotEid}`);
    }
    if (Children.entitiesCount[slotEid] === 0) {
        return 0;
    }
    return Children.entitiesIds.get(slotEid, 0);
}
