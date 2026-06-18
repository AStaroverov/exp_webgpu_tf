import { query, removeComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getComponents } from "renderer/src/ECS/utils.ts";
import { createExpirySubComponent } from "../Components/lib/createExpirySubComponent.ts";

/**
 * Ticks every component composed from `createExpirySubComponent` down by the
 * frame delta and removes it at 0. The components are discovered through the
 * sub-component registry — a new expiring status only spreads the
 * sub-component, no wiring here.
 */
export function createExpirySystem({ world } = GameDI) {
  const components = getComponents(world, createExpirySubComponent);

  return (delta: number) => {
    for (let c = 0; c < components.length; c++) {
      const component = components[c];
      const eids = query(world, [component]);

      // Backwards: removeComponent swap-removes inside the query's dense array.
      for (let i = eids.length - 1; i >= 0; i--) {
        const eid = eids[i];

        if (component.tick(eid, delta)) {
          removeComponent(world, eid, component);
        }
      }
    }
  };
}
