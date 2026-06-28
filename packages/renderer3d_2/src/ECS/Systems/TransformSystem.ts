import { mat4 } from "gl-matrix";
import { getRenderComponents, type RenderWorld } from "../world.ts";
import { query } from "bitecs";
type ChildrenLike = {
  entitiesCount: { get(eid: number): number };
  entitiesIds: { get(eid: number, i: number): number };
};

export function createTransformSystem(world: RenderWorld, Children: ChildrenLike) {
  const { GlobalTransform, LocalTransform } = getRenderComponents(world);
  return function execMainTransformSystem() {
    {
      const entities = query(world, [LocalTransform, GlobalTransform]);

      for (let i = 0; i < entities.length; i++) {
        const id = entities[i];
        const local = LocalTransform.matrix.getBatch(id);
        GlobalTransform.matrix.setBatch(id, local);
      }
    }

    {
      const entities = query(world, [GlobalTransform, Children]);
      const tmp = mat4.create();

      for (let i = 0; i < entities.length; i++) {
        const id = entities[i];
        const globalParent = GlobalTransform.matrix.getBatch(id);
        const childrenCount = Children.entitiesCount.get(id);
        for (let j = 0; j < childrenCount; j++) {
          const childId = Children.entitiesIds.get(id, j);
          const localChild = LocalTransform.matrix.getBatch(childId);
          mat4.multiply(tmp, globalParent, localChild);
          GlobalTransform.matrix.setBatch(childId, tmp);
        }
      }
    }
  };
}
