import { type EntityId, hasComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function isSlot(eid: EntityId, { world } = GameDI): boolean {
  const { Slot, Children } = getGameComponents(world);
  const hasSlot = hasComponent(world, eid, Slot);
  const hasChildren = hasComponent(world, eid, Children);
  if (hasSlot && !hasChildren) {
    throw new Error(`Children component not found for slot entity ${eid}`);
  }
  return hasSlot && hasChildren;
}

export function isSlotFilled(slotEid: EntityId, { world } = GameDI): boolean {
  const { Children } = getGameComponents(world);
  return Children.entitiesCount.get(slotEid) > 0;
}

export function isSlotEmpty(slotEid: EntityId, { world } = GameDI): boolean {
  const { Children } = getGameComponents(world);
  return Children.entitiesCount.get(slotEid) === 0;
}

export function getSlotFillerEid(slotEid: EntityId, { world } = GameDI): EntityId {
  const { Children } = getGameComponents(world);
  return Children.entitiesIds.get(slotEid, 0);
}
