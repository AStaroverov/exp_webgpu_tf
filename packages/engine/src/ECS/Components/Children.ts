import { addComponent, World } from "bitecs";
import { defineComponent } from "../../../../common/src/component.ts";

const MAX_CHILDREN = 256;

export const createChildrenComponent = defineComponent((Children, ctx) => {
  const entitiesCount = ctx.table.flat(Float64Array);
  const entitiesIds = ctx.table.nested(Float64Array, MAX_CHILDREN);

  return {
    entitiesCount,
    entitiesIds,

    addComponent(world: World, eid: number) {
      addComponent(world, eid, Children);
      entitiesCount.set(eid, 0);
    },

    addChild(parentEid: number, childEid: number) {
      const len = entitiesCount.get(parentEid);
      if (len >= MAX_CHILDREN) throw new Error("Children: max reached");
      entitiesIds.set(parentEid, len, childEid);
      entitiesCount.set(parentEid, len + 1);
    },

    removeChild(parentEid: number, childEid: number) {
      const len = entitiesCount.get(parentEid);
      let index = -1;
      for (let i = 0; i < len; i++) {
        if (entitiesIds.get(parentEid, i) === childEid) {
          index = i;
          break;
        }
      }
      if (index === -1) return;
      for (let i = index; i < len - 1; i++) {
        entitiesIds.set(parentEid, i, entitiesIds.get(parentEid, i + 1));
      }
      entitiesCount.set(parentEid, len - 1);
    },
  };
});
