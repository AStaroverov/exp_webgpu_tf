import { mat4 } from "gl-matrix";
import { getRenderComponents, type RenderWorld } from "../world.ts";
import { query } from "bitecs";
type ChildrenLike = {
  entitiesCount: { get(eid: number): number };
  entitiesIds: { get(eid: number, i: number): number };
};

export function createTransformSystem(world: RenderWorld, Children: ChildrenLike, rootEid: number) {
  const { GlobalTransform, LocalTransform } = getRenderComponents(world);
  const tmp = mat4.create();
  // The scene graph has one explicit root (the SceneNode). We DFS the Children tree from it so a
  // parent's GlobalTransform is finalized before its children read it — correct for any depth,
  // with no root discovery and no reliance on entity-id order. rootEid < 0 ⇒ no graph (skip).
  const stack: number[] = [];
  return function execMainTransformSystem() {
    {
      const entities = query(world, [LocalTransform, GlobalTransform]);

      for (let i = 0; i < entities.length; i++) {
        const id = entities[i];
        const local = LocalTransform.matrix.getBatch(id);
        GlobalTransform.matrix.setBatch(id, local);
      }
    }

    if (rootEid >= 0) {
      stack.length = 0;
      stack.push(rootEid);
      while (stack.length > 0) {
        const id = stack.pop() as number;
        const globalParent = GlobalTransform.matrix.getBatch(id);
        const childrenCount = Children.entitiesCount.get(id);
        for (let j = 0; j < childrenCount; j++) {
          const childId = Children.entitiesIds.get(id, j);
          const localChild = LocalTransform.matrix.getBatch(childId);
          mat4.multiply(tmp, globalParent, localChild);
          GlobalTransform.matrix.setBatch(childId, tmp);
          if (Children.entitiesCount.get(childId) > 0) stack.push(childId);
        }
      }
    }
  };
}
