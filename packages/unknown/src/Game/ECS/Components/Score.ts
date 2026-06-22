import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Per-entity running score (points). Pure data: `add` accumulates, `get` reads.
 * Attached to whichever vehicle keeps a score (the demo gives it to the human's
 * tank); the hit pipeline credits it from enemy damage dealt, and the UI reads
 * it each frame.
 */
export const createScoreComponent = defineComponent((Score, ctx) => {
  const points = ctx.table.flat(Float64Array);
  return {
    points,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, Score);
    },
    add(eid: EntityId, amount: number) {
      points.set(eid, points.get(eid) + amount);
    },
    get(eid: EntityId): number {
      return points.get(eid);
    },
  };
});
