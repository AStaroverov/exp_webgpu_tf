import { addComponent, removeComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

export enum WheelPosition {
  FrontLeft = 0,
  FrontRight = 1,
  RearLeft = 2,
  RearRight = 3,
}

export const createWheelComponent = defineComponent((Wheel) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, Wheel);
  },
}));

export const createWheelSteerableComponent = defineComponent((WheelSteerable, ctx) => {
  const maxSteeringAngle = ctx.table.flat(Float64Array);
  const steeringSpeed = ctx.table.flat(Float64Array);
  return {
    maxSteeringAngle,
    steeringSpeed,
    addComponent(
      world: World,
      eid: EntityId,
      maxAngle: number = Math.PI / 6,
      speed: number = Math.PI * 2,
    ) {
      addComponent(world, eid, WheelSteerable);
      maxSteeringAngle.set(eid, maxAngle);
      steeringSpeed.set(eid, speed);
    },
    removeComponent(world: World, eid: EntityId) {
      removeComponent(world, eid, WheelSteerable);
    },
  };
});

export const createWheelDriveComponent = defineComponent((WheelDrive) => ({
  addComponent(world: World, eid: EntityId) {
    addComponent(world, eid, WheelDrive);
  },
  removeComponent(world: World, eid: EntityId) {
    removeComponent(world, eid, WheelDrive);
  },
}));
