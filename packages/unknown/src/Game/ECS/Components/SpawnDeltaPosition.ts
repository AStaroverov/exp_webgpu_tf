import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Projectile spawn point as an offset in the owner's LOCAL space (e.g. the gun
 * tip relative to the turret pivot). Shared by every weapon kind — bullets and
 * stream particles both spawn at owner transform × this delta.
 */
export const createSpawnDeltaPositionComponent = defineComponent((SpawnDeltaPosition) => {
  const position = NestedArray.f32(2, delegate.defaultSize);
  return {
    position,
    addComponent(world: World, eid: EntityId, x: number, y: number) {
      addComponent(world, eid, SpawnDeltaPosition);
      position.set(eid, 0, x);
      position.set(eid, 1, y);
    },
  };
});
