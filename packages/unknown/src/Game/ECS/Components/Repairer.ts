import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

/**
 * Marks a tank that repairs itself by salvaging ground scrap: when it sits near a
 * cluster of `Salvage` debris, `createRepairSystem` consumes them to refill one
 * slot. Holds the live cooldown (remaining ms before the next salvage); the radius
 * / pieces-needed / cooldown-length tunables are constants at the system's use
 * site. Only the human player's tank carries it (set in `setupDemoWorld`).
 */
export const createRepairerComponent = defineComponent((Repairer, { table }) => {
  const cooldown = table.flat(Float64Array);
  return {
    cooldown,
    addComponent(world: World, eid: number) {
      addComponent(world, eid, Repairer); // cooldown zero-filled → ready at once
    },
  };
});
