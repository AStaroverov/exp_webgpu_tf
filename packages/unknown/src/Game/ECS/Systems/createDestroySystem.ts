import { GameDI } from "../../DI/GameDI.ts";
import { query } from "bitecs";
import { recursiveTypicalRemoveEntity, typicalRemoveEntity } from "../Utils/typicalRemoveEntity.ts";
import { getGameComponents } from "../createGameWorld.ts";

export function createDestroySystem({ world } = GameDI) {
  const { Destroy } = getGameComponents(world);

  return () => {
    const eids = query(world, [Destroy]);

    for (let i = 0; i < eids.length; i++) {
      const eid = eids[i];
      const recursive = Destroy.recursive[eid] === 1;
      recursive ? recursiveTypicalRemoveEntity(eid) : typicalRemoveEntity(eid);
    }
  };
}
