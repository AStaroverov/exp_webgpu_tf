import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Marks the tank a human drives directly (keyboard + mouse), bypassing the
 * action queue. Its presence is the boolean: the stand-in/AI decision driver
 * queries it out so it never enqueues actions onto a manually-driven tank
 * (queued actions would fight the per-frame controller writes — see
 * `createManualControl`). Nothing else needs to type-check "is this the player".
 */
export const createPlayerControlledComponent = defineComponent((PlayerControlled) => {
  return {
    addComponent(world: World, eid: number) {
      addComponent(world, eid, PlayerControlled);
    },
  };
});
