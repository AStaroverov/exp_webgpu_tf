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
      const origin = DestroyByDistance.origin.getBatch(eid);
      const dx = position[0] - origin[0];
      const dy = position[1] - origin[1];

      if (dx * dx + dy * dy > DestroyByDistance.maxDistanceSq[eid]) {
        scheduleRemoveEntity(eid);
      }
    }
  };
}
