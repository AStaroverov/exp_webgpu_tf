import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * Projectile spawn point as an offset in the owner's LOCAL space (e.g. the gun
 * tip relative to the turret pivot). Shared by every weapon kind — bullets and
 * stream particles both spawn at owner transform × this delta.
 */
export const createSpawnDeltaPositionComponent = defineComponent((SpawnDeltaPosition, ctx) => {
  const position = ctx.table.nested(Float32Array, 2);
  return {
    position,
    addComponent(world: World, eid: EntityId, x: number, y: number) {
      addComponent(world, eid, SpawnDeltaPosition);
      position.set(eid, 0, x);
      position.set(eid, 1, y);
    },
  };
});
