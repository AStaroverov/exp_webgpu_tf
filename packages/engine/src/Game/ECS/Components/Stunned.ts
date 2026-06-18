import { defineComponent } from "renderer/src/ECS/utils.ts";
import { createExpirySubComponent } from "./lib/createExpirySubComponent.ts";

/**
 * Presence = "this vehicle is fully disabled". `remainingMs` is the live
 * countdown; the gating sites (track control, turret rotation, both weapon
 * spawners) read only membership. Every Emp-kind damage event refreshes it
 * (max, not stack); `createExpirySystem` ticks it down and removes the
 * component at 0.
 */
export const createStunnedComponent = defineComponent((Stunned, ctx) => {
  return {
    ...createExpirySubComponent(Stunned, ctx),
  };
});
