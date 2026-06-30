import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../common/src/component.ts";

// Desired LINEAR velocity for a body, set on the PRODUCER (main) world. It is pure
// per-entity state — no physics here. createApplyVelocitySystem queries [RigidBodyState,
// Velocity] and emits a SET_VELOCITY op each frame; the worker applies it with setLinvel.
// Main-private (not a shared bridge column): only the producer system reads it.
export const createVelocityComponent = defineComponent((Velocity, ctx) => {
  const x = ctx.table.flat(Float64Array);
  const y = ctx.table.flat(Float64Array);
  const z = ctx.table.flat(Float64Array);
  return {
    x,
    y,
    z,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, Velocity);
    },
    set(eid: number, vx: number, vy: number, vz: number) {
      x.set(eid, vx);
      y.set(eid, vy);
      z.set(eid, vz);
    },
    reset(eid: number) {
      x.set(eid, 0);
      y.set(eid, 0);
      z.set(eid, 0);
    },
  };
});
