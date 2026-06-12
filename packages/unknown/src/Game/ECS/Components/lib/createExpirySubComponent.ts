import { addComponent, EntityId, hasComponent } from "bitecs";
import { delegate } from "../../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../../renderer/src/utils.ts";
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
export const createExpirySubComponent = defineSubComponent((Component, { world }) => {
  const durationMs = TypedArray.f64(delegate.defaultSize);
  const remainingMs = TypedArray.f64(delegate.defaultSize);

  return {
    durationMs,
    remainingMs,
    tick(eid: EntityId, delta: number): boolean {
      remainingMs[eid] -= delta;
      return remainingMs[eid] <= 0;
    },
    refresh(eid: EntityId, ms: number) {
      if (!hasComponent(world, eid, Component)) {
        addComponent(world, eid, Component);
        remainingMs[eid] = 0;
        durationMs[eid] = 0;
      }
      remainingMs[eid] = Math.max(remainingMs[eid], ms);
      durationMs[eid] = Math.max(durationMs[eid], ms);
    },
    getRemainingFraction(eid: EntityId): number {
      return remainingMs[eid] / durationMs[eid];
    },
  };
});
