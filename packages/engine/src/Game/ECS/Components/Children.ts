import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";

const MAX_CHILDREN = 1000;

export const createChildrenComponent = defineComponent((Children, ctx) => {
  const entitiesCount = ctx.table.flat(Float64Array);
  const entitiesIds = ctx.table.nested(Float64Array, MAX_CHILDREN);

  function removeAllChildren(entity: number) {
    const len = entitiesCount.get(entity);
    for (let i = 0; i < len; i++) {
      entitiesIds.set(entity, i, 0);
    }
    entitiesCount.set(entity, 0);
  }

  return {
    entitiesCount,
    entitiesIds,

    addComponent(world: World, eid: number, ids: number[] | Float64Array = []) {
      addComponent(world, eid, Children);
      removeAllChildren(eid);
      entitiesCount.set(eid, ids.length);
      entitiesIds.setBatch(eid, ids);
    },

    addChildren(entity: number, child: number) {
      const len = entitiesCount.get(entity);
      if (len >= MAX_CHILDREN) {
        throw new Error("Max children reached");
      }
      entitiesIds.set(entity, len, child);
      entitiesCount.set(entity, len + 1);
    },

    removeAllChildren,

    removeChild(parentEid: number, childEid: number) {
      const len = entitiesCount.get(parentEid);

      if (len === 0) {
        return removeAllChildren(parentEid);
      }

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
      entitiesIds.set(parentEid, len - 1, 0);
      entitiesCount.set(parentEid, len - 1);
    },
  };
});
