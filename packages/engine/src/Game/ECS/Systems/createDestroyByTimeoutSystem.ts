import { GameDI } from "../../DI/GameDI.ts";
import { query } from "bitecs";
import { scheduleRemoveEntity } from "../Utils/typicalRemoveEntity.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createDestroyByTimeoutSystem({ world } = GameDI) {
  const { DestroyByTimeout } = getGameComponents(world);

  return (delta: number) => {
    const eids = query(world, [DestroyByTimeout]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];

      DestroyByTimeout.updateTimeout(eid, delta);

      if (DestroyByTimeout.timeout.get(eid) <= 0) {
        scheduleRemoveEntity(eid);
      }
    }
  };
}
