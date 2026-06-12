import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Sinusoidal steering for free-flying bodies (stream particles): each tick the
 * velocity vector is rotated by `sin(age · frequency + phase) · angularSpeed`,
 * so the path meanders along a smooth curve instead of a straight ray.
 * `phase` is seeded at spawn — every particle curves its own way, reproducibly.
 */
export const createWanderComponent = defineComponent((Wander, ctx) => {
  /** Per-entity phase offset in radians — decorrelates particles */
  const phase = ctx.table.flat(Float32Array);
  /** Steering oscillation frequency in rad/ms */
  const frequency = ctx.table.flat(Float32Array);
  /** Peak turn rate in rad/s */
  const angularSpeed = ctx.table.flat(Float32Array);
  /** Time since spawn in ms — the argument of the steering sine */
  const ageMs = ctx.table.flat(Float64Array);
  return {
    phase,
    frequency,
    angularSpeed,
    ageMs,
    addComponent(world: World, eid: EntityId, ph: number, freq: number, turn: number) {
      addComponent(world, eid, Wander);
      phase.set(eid, ph);
      frequency.set(eid, freq);
      angularSpeed.set(eid, turn);
    },
  };
});
