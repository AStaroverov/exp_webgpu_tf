import { addComponent, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

const MAX_CHILDREN = 1000;

export const createChildrenComponent = defineComponent((Children) => {
  const entitiesCount = new Float64Array(delegate.defaultSize);
  const entitiesIds = NestedArray.f64(MAX_CHILDREN, delegate.defaultSize);

  function removeAllChildren(entity: number) {
    entitiesCount[entity] = 0;
    entitiesIds.getBatch(entity).fill(0);
  }

  return {
    entitiesCount,
    entitiesIds,

    addComponent(world: World, eid: number, ids: number[] | Float64Array = []) {
      addComponent(world, eid, Children);
      entitiesCount[eid] = ids.length;
      entitiesIds.getBatch(eid).fill(0);
      entitiesIds.setBatch(eid, ids);
    },

    addChildren(entity: number, child: number) {
      const len = entitiesCount[entity];
      if (len >= MAX_CHILDREN) {
        throw new Error("Max children reached");
      }
      entitiesIds.set(entity, len, child);
      entitiesCount[entity] += 1;
    },

    removeAllChildren,

    removeChild(parentEid: number, childEid: number) {
      const children = entitiesIds.getBatch(parentEid);
      const len = entitiesCount[parentEid];

      if (len === 0) {
        return removeAllChildren(parentEid);
      }

      const index = children.subarray(0, len).indexOf(childEid);
      if (index === -1) return;

      entitiesCount[parentEid] -= 1;
      children.set(children.subarray(index + 1, len), index);
      children[entitiesCount[parentEid]] = 0;
    },
  };
});
