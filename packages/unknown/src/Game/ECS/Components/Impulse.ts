import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export const createImpulseComponent = defineComponent((Impulse) => {
  const x = TypedArray.f64(delegate.defaultSize);
  const y = TypedArray.f64(delegate.defaultSize);
  return {
    x,
    y,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, Impulse);
      x[eid] = 0;
      y[eid] = 0;
    },
    add(eid: EntityId, ax: number, ay: number) {
      x[eid] += ax;
      y[eid] += ay;
    },
    set(eid: EntityId, sx: number, sy: number) {
      x[eid] = sx;
      y[eid] = sy;
    },
    reset(eid: EntityId) {
      x[eid] = 0;
      y[eid] = 0;
    },
    hasImpulse(eid: EntityId): boolean {
      return x[eid] !== 0 || y[eid] !== 0;
    },
  };
});

export const createTorqueImpulseComponent = defineComponent((TorqueImpulse) => {
  const value = TypedArray.f64(delegate.defaultSize);
  return {
    value,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, TorqueImpulse);
      value[eid] = 0;
    },
    add(eid: EntityId, torque: number) {
      value[eid] += torque;
    },
    set(eid: EntityId, torque: number) {
      value[eid] = torque;
    },
    reset(eid: EntityId) {
      value[eid] = 0;
    },
    hasImpulse(eid: EntityId): boolean {
      return value[eid] !== 0;
    },
  };
});

const MAX_IMPULSE_POINTS = 4;

export const createImpulseAtPointComponent = defineComponent((ImpulseAtPoint) => {
  const impulseX = TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS);
  const impulseY = TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS);
  const pointX = TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS);
  const pointY = TypedArray.f64(delegate.defaultSize * MAX_IMPULSE_POINTS);
  const count = TypedArray.i8(delegate.defaultSize);
  return {
    impulseX,
    impulseY,
    pointX,
    pointY,
    count,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, ImpulseAtPoint);
      count[eid] = 0;
    },
    add(eid: EntityId, ix: number, iy: number, wx: number, wy: number) {
      const idx = count[eid];
      if (idx >= MAX_IMPULSE_POINTS) return;
      const offset = eid * MAX_IMPULSE_POINTS + idx;
      impulseX[offset] = ix;
      impulseY[offset] = iy;
      pointX[offset] = wx;
      pointY[offset] = wy;
      count[eid] = idx + 1;
    },
    get(eid: EntityId, index: number): [number, number, number, number] {
      const offset = eid * MAX_IMPULSE_POINTS + index;
      return [impulseX[offset], impulseY[offset], pointX[offset], pointY[offset]];
    },
    reset(eid: EntityId) {
      count[eid] = 0;
    },
    hasImpulse(eid: EntityId): boolean {
      return count[eid] > 0;
    },
  };
});
