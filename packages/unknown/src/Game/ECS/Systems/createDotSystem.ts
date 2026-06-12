import { query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

/**
 * Records this tick's damage for every part under a `Dot`. Damage goes through
 * `Hitable.hit$` like every other damage type — the hitable pass applies it and
 * triggers the kind's specialty (accepting a harmless one-frame lag). The
 * self-`eid` source attributes nothing in `LastHitters` (same team);
 * attribution happened at stamp time from the projectile. The countdown and
 * removal are the expiry sub-component's business (`createExpirySystem`).
 */
export function createDotSystem({ world } = GameDI) {
  const { Dot, Hitable } = getGameComponents(world);

  return (delta: number) => {
    const eids = query(world, [Dot, Hitable]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];
      Hitable.hit$(eid, eid, (Dot.dps[eid] * delta) / 1000, Dot.kind[eid]);
    }
  };
}
