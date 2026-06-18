import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

/**
 * Animation track for a `LightEmitter`: the emitter's PEAK params — its values
 * at progress 0. Presence (together with `Progress` as the clock) = "this light
 * decays over its lifetime": the animation system rescales the live emitter
 * from these each frame, so the live value never accumulates error.
 */
export const createLightEmitterAnimationComponent = defineComponent(
  (LightEmitterAnimation, ctx) => {
    const intensity = ctx.table.flat(Float64Array);
    const radius = ctx.table.flat(Float64Array);
    return {
      intensity,
      radius,
      addComponent(world: World, eid: EntityId, i: number, r = 0) {
        addComponent(world, eid, LightEmitterAnimation);
        intensity.set(eid, i);
        radius.set(eid, r);
      },
    };
  },
);
