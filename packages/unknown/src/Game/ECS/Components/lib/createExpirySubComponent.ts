import { addComponent, hasComponent } from "bitecs";
import type { EntityId } from "bitecs";
import { defineSubComponent } from "../../../../../../renderer/src/ECS/utils.ts";

export type Expiry = ReturnType<typeof createExpirySubComponent>;

/**
 * A countdown, spread into a duration-status component
 * (`{ ...createExpirySubComponent(Ref, ctx) }`). Purely about time — what the
 * status means while present is the component's own business.
 *
 * - `refresh(eid, durationMs)` adds the component on first touch and extends
 *   the countdown to at least `durationMs` (max, not stack);
 * - `createExpirySystem` ticks it down and removes the component at 0;
 * - `getRemainingFraction(eid)` ∈ [0, 1] — for visuals fading with time left.
 */
export const createExpirySubComponent = defineSubComponent((Component, { world, table }) => {
  const durationMs = table.flat(Float64Array);
  const remainingMs = table.flat(Float64Array);

  return {
    durationMs,
    remainingMs,
    tick(eid: EntityId, delta: number): boolean {
      const left = remainingMs.get(eid) - delta;
      remainingMs.set(eid, left);
      return left <= 0;
    },
    refresh(eid: EntityId, ms: number) {
      if (!hasComponent(world, eid, Component)) {
        addComponent(world, eid, Component); // creates the row zeroed
      }
      remainingMs.set(eid, Math.max(remainingMs.get(eid), ms));
      durationMs.set(eid, Math.max(durationMs.get(eid), ms));
    },
    getRemainingFraction(eid: EntityId): number {
      return remainingMs.get(eid) / durationMs.get(eid);
    },
  };
});
