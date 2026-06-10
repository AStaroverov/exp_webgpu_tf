import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { addComponent, EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { SlotPartType } from "./SlotConfig.ts";

export const createSlotComponent = defineComponent((Slot) => {
  const anchorX = TypedArray.f64(delegate.defaultSize);
  const anchorY = TypedArray.f64(delegate.defaultSize);
  const width = TypedArray.f64(delegate.defaultSize);
  const height = TypedArray.f64(delegate.defaultSize);
  const partType = TypedArray.i8(delegate.defaultSize);

  return {
    anchorX,
    anchorY,
    width,
    height,
    partType,

    addComponent(
      world: World,
      eid: EntityId,
      x: number,
      y: number,
      w: number,
      h: number,
      type: SlotPartType,
    ): EntityId {
      addComponent(world, eid, Slot);
      partType[eid] = type;
      anchorX[eid] = x;
      anchorY[eid] = y;
      width[eid] = w;
      height[eid] = h;
      return eid;
    },
  };
});
