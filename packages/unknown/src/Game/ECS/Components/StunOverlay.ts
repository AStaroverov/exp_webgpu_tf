import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * Render-side bookkeeping on a stunned vehicle: which overlay entity carries
 * its EmpOverlay VFX + ground glow. Relation by eid — sentinel `0` = none, the
 * eid is verified with `entityExists` before use (project rule). Lives only
 * while `createStunArcsSystem` keeps the overlay alive.
 */
export const createStunOverlayComponent = defineComponent((StunArcs, ctx) => {
  const overlayEid = ctx.table.flat(Uint32Array);
  return {
    overlayEid,
    addComponent(world: World, eid: number, overlay: number) {
      addComponent(world, eid, StunArcs);
      overlayEid.set(eid, overlay);
    },
  };
});
