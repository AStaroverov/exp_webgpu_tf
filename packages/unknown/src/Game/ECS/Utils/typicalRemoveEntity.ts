import { hasComponent, removeEntity } from "bitecs";
import { removeRigidShape } from "../../Physical/createRigid.ts";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function scheduleRemoveEntity(eid: number, recursive = true, { world } = GameDI) {
  const { Destroy } = getGameComponents(world);
  Destroy.addComponent(world, eid, recursive);
}

export function typicalRemoveEntity(
  eid: number,
  disconnect = true,
  { world, physicalWorld } = GameDI,
) {
  const { RigidBodyRef, Parent, Children, CompoundPart } = getGameComponents(world);

  if (hasComponent(world, eid, RigidBodyRef)) {
    removeRigidShape(eid);
    RigidBodyRef.clear(eid);
  }

  if (hasComponent(world, eid, CompoundPart)) {
    const collider = physicalWorld.getCollider(CompoundPart.colliderHandle.get(eid));
    if (collider?.parent()) physicalWorld.removeCollider(collider, true);
  }

  if (
    disconnect &&
    hasComponent(world, eid, Parent) &&
    hasComponent(world, Parent.id.get(eid), Children)
  ) {
    Children.removeChild(Parent.id.get(eid), eid);
  }

  removeEntity(world, eid);
}

export function recursiveTypicalRemoveEntity(eid: number, isRoot = true, { world } = GameDI) {
  const { Children } = getGameComponents(world);
  if (hasComponent(world, eid, Children)) {
    for (let i = 0; i < Children.entitiesCount.get(eid); i++) {
      recursiveTypicalRemoveEntity(Children.entitiesIds.get(eid, i), false);
    }
    Children.entitiesCount.set(eid, 0);
  }

  typicalRemoveEntity(eid, isRoot);
}
