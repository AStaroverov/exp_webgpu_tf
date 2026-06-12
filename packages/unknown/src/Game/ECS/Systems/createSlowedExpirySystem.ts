import { query, removeComponent } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { FrostSlowConfig } from "../../Config/weapons.ts";

/**
 * Thaws `Slowed`: freeze contributions are added by the hitable pipeline
 * (Frost-kind damage), this system recovers `slowMul` by a fixed step per
 * tick and removes the component at full speed. The slow itself is read
 * inline at the speed sites (track control, turret rotation).
 */
export function createSlowedExpirySystem({ world } = GameDI) {
  const { Slowed } = getGameComponents(world);

  return (_delta: number) => {
    const eids = query(world, [Slowed]);

    // Backwards: removeComponent swap-removes inside the query's dense array.
    for (let i = eids.length - 1; i >= 0; i--) {
      const eid = eids[i];

      const next = Slowed.slowMul.get(eid) - FrostSlowConfig.thawPerTick;
      Slowed.slowMul.set(eid, next);
      if (next <= 0) {
        removeComponent(world, eid, Slowed);
      }
    }
  };
}
