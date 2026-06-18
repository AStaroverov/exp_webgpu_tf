import { addComponent, observe, onRemove, removeComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * A vehicle armor part whose collider lives ON the parent body (hull or turret)
 * instead of being its own rigid body bolted on with a fixed joint. This is the
 * "compound collider" representation: many parts share one body, so the solver
 * sees zero extra bodies and zero fixed joints for them.
 *
 * The part entity keeps its own per-part components (`Hitable`, `Damagable`,
 * `VehiclePart`, render shape) — only the PHYSICS representation is folded into
 * the owner body. The link back from a collider to its part eid is kept here as
 * a plain map (`colliderHandle -> eid`), so contact/sensor drains can attribute
 * a hit to the exact part even though all part colliders report the same parent
 * body. The maps are plain JS (not table columns) so cleanup is independent of
 * the component table's row lifecycle.
 */
const colliderToEid = new Map<number, number>();
const eidToCollider = new Map<number, number>();

export function getEntityIdByColliderId(colliderHandle: number): number | undefined {
  return colliderToEid.get(colliderHandle);
}

export const createCompoundPartComponent = defineComponent((CompoundPart, ctx) => {
  const ownerEid = ctx.table.flat(Float64Array);
  const colliderHandle = ctx.table.flat(Float64Array);
  const anchorX = ctx.table.flat(Float64Array);
  const anchorY = ctx.table.flat(Float64Array);

  const unlink = (eid: number) => {
    const handle = eidToCollider.get(eid);
    if (handle !== undefined) {
      colliderToEid.delete(handle);
      eidToCollider.delete(eid);
    }
  };

  // Safety net: if the entity is removed without an explicit removeComponent
  // (e.g. recursive entity teardown), still drop the collider->eid mapping.
  observe(ctx.world, onRemove(CompoundPart), unlink);

  return {
    ownerEid,
    colliderHandle,
    anchorX,
    anchorY,
    addComponent(
      world: World,
      eid: EntityId,
      owner: number,
      handle: number,
      ax: number,
      ay: number,
    ) {
      addComponent(world, eid, CompoundPart);
      ownerEid.set(eid, owner);
      colliderHandle.set(eid, handle);
      anchorX.set(eid, ax);
      anchorY.set(eid, ay);
      colliderToEid.set(handle, eid);
      eidToCollider.set(eid, handle);
    },
    removeComponent(world: World, eid: EntityId) {
      unlink(eid);
      removeComponent(world, eid, CompoundPart);
    },
    dispose() {
      colliderToEid.clear();
      eidToCollider.clear();
    },
  };
});
