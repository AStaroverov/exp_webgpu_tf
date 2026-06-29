import { hasComponent, removeEntity } from "bitecs";
import { getEngineComponents, type EngineWorld } from "./createEngineWorld.ts";

// Remove an entity and its whole subtree (depth-first). Reads Children to walk down;
// eids never recycle (shared counter), so the descendants gathered are unambiguous.
export function removeEntityTree(world: EngineWorld, root: number): void {
  const { Children } = getEngineComponents(world);
  if (hasComponent(world, root, Children)) {
    const count = Children.entitiesCount.get(root);
    for (let i = 0; i < count; i++) {
      removeEntityTree(world, Children.entitiesIds.get(root, i));
    }
  }
  removeEntity(world, root);
}
