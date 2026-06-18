import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { scheduleRemoveEntity } from "../../Utils/typicalRemoveEntity.ts";
import { MAX_TREAD_MARKS } from "../../Entities/TreadMark.ts";

/**
 * Caps the live tread-mark population at `MAX_TREAD_MARKS`. Emission is per
 * caterpillar-segment AND per wheel — hundreds of independent emitters across a
 * fleet — so during turns the spawn rate (∝ emitters × turn-rate) balloons the
 * count and overflows the renderer's instance buffer. This trims the overflow
 * back to the ceiling instead of capping at the spawn site (the entity factory
 * stays dumb; the population policy lives here, in a system).
 *
 * The query's dense set is append-at-back / oldest-expire-from-front, so the
 * leading entries are the oldest marks — evicting from the front drops the
 * faintest (closest-to-expiry) trail first. Marks are decals, so approximate
 * age ordering is fine.
 */
export function createLimitTreadMarksSystem({ world } = GameDI) {
  const { TreadMark } = getGameComponents(world);

  return () => {
    const eids = query(world, [TreadMark]);
    const excess = eids.length - MAX_TREAD_MARKS;
    if (excess <= 0) return;

    for (let i = 0; i < excess; i++) {
      scheduleRemoveEntity(eids[i]);
    }
  };
}
