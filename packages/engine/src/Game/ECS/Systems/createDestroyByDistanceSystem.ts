import { GameDI } from "../../DI/GameDI.ts";
import { query } from "bitecs";
import { scheduleRemoveEntity } from "../Utils/typicalRemoveEntity.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createDestroyByDistanceSystem({ world } = GameDI) {
  const { DestroyByDistance, RigidBodyState } = getGameComponents(world);

  return () => {
    const eids = query(world, [DestroyByDistance, RigidBodyState]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];
      const position = RigidBodyState.position.getBatch(eid);
      const ox = DestroyByDistance.origin.get(eid, 0);
      const oy = DestroyByDistance.origin.get(eid, 1);
      const dx = position[0] - ox;
      const dy = position[1] - oy;

      if (dx * dx + dy * dy > DestroyByDistance.maxDistanceSq.get(eid)) {
        scheduleRemoveEntity(eid);
      }
    }
  };
}
