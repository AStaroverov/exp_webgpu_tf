import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createImpulseComponent = defineComponent((Impulse, ctx) => {
  const x = ctx.table.flat(Float64Array);
  const y = ctx.table.flat(Float64Array);
  return {
    x,
    y,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, Impulse);
    },
    add(eid: EntityId, ax: number, ay: number) {
      x.set(eid, x.get(eid) + ax);
      y.set(eid, y.get(eid) + ay);
    },
    set(eid: EntityId, sx: number, sy: number) {
      x.set(eid, sx);
      y.set(eid, sy);
    },
    reset(eid: EntityId) {
      x.set(eid, 0);
      y.set(eid, 0);
    },
    hasImpulse(eid: EntityId): boolean {
      return x.get(eid) !== 0 || y.get(eid) !== 0;
    },
  };
});

export const createTorqueImpulseComponent = defineComponent((TorqueImpulse, ctx) => {
  const value = ctx.table.flat(Float64Array);
  return {
    value,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, TorqueImpulse);
    },
    add(eid: EntityId, torque: number) {
      value.set(eid, value.get(eid) + torque);
    },
    set(eid: EntityId, torque: number) {
      value.set(eid, torque);
    },
    reset(eid: EntityId) {
      value.set(eid, 0);
    },
    hasImpulse(eid: EntityId): boolean {
      return value.get(eid) !== 0;
    },
  };
});

const MAX_IMPULSE_POINTS = 4;

export const createImpulseAtPointComponent = defineComponent((ImpulseAtPoint, ctx) => {
  const impulseX = ctx.table.nested(Float64Array, MAX_IMPULSE_POINTS);
  const impulseY = ctx.table.nested(Float64Array, MAX_IMPULSE_POINTS);
  const pointX = ctx.table.nested(Float64Array, MAX_IMPULSE_POINTS);
  const pointY = ctx.table.nested(Float64Array, MAX_IMPULSE_POINTS);
  const count = ctx.table.flat(Int8Array);
  return {
    impulseX,
    impulseY,
    pointX,
    pointY,
    count,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, ImpulseAtPoint);
    },
    add(eid: EntityId, ix: number, iy: number, wx: number, wy: number) {
      const idx = count.get(eid);
      if (idx >= MAX_IMPULSE_POINTS) return;
      impulseX.set(eid, idx, ix);
      impulseY.set(eid, idx, iy);
      pointX.set(eid, idx, wx);
      pointY.set(eid, idx, wy);
      count.set(eid, idx + 1);
    },
    get(eid: EntityId, index: number): [number, number, number, number] {
      return [
        impulseX.get(eid, index),
        impulseY.get(eid, index),
        pointX.get(eid, index),
        pointY.get(eid, index),
      ];
    },
    reset(eid: EntityId) {
      count.set(eid, 0);
    },
    hasImpulse(eid: EntityId): boolean {
      return count.get(eid) > 0;
    },
  };
});
