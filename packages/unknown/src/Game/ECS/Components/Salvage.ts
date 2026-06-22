import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Marks a torn-off tank part lying on the ground as salvageable scrap. Tagged in
 * `tearOffTankPart` so every piece of debris (from damage or a kill) carries it;
 * presence is the "this is collectable scrap" query. A `Repairer` tank driving
 * over a cluster of these consumes them to refill a slot (see `createRepairSystem`).
 */
export const createSalvageComponent = defineComponent((Salvage) => {
  return {
    addComponent(world: World, eid: number) {
      addComponent(world, eid, Salvage);
    },
  };
});
