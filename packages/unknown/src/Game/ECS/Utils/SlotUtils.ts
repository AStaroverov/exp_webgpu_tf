import { EntityId, hasComponent } from 'bitecs';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';

export function isSlot(world: RenderGameWorld, eid: EntityId): boolean {
    const { Slot, Children } = getRenderWorldComponents(world);
    const hasSlot = hasComponent(world, eid, Slot);
    const hasChildren = hasComponent(world, eid, Children);
    if (hasSlot && !hasChildren) {
        throw new Error(`Children component not found for slot entity ${eid}`);
    }
    return hasSlot && hasChildren;
}

export function isSlotFilled(world: RenderGameWorld, slotEid: EntityId): boolean {
    const { Children } = getRenderWorldComponents(world);
    return Children.entitiesCount[slotEid] > 0;
}

export function isSlotEmpty(world: RenderGameWorld, slotEid: EntityId): boolean {
    const { Children } = getRenderWorldComponents(world);
    return Children.entitiesCount[slotEid] === 0;
}

export function getSlotFillerEid(world: RenderGameWorld, slotEid: EntityId): EntityId {
    const { Children } = getRenderWorldComponents(world);
    return Children.entitiesIds.get(slotEid, 0);
}
