import {
  GlobalTransform,
  setMatrixRotateZ,
  setMatrixTranslate,
} from "../../../../../renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../DI/GameDI.ts";
import { query } from "bitecs";
import { getGameComponents } from "../createGameWorld.ts";

export function createApplyRigidBodyToTransformSystem({ world } = GameDI) {
  const { RigidBodyRef, RigidBodyState } = getGameComponents(world);

  return function () {
    const entities = query(world, [GlobalTransform, RigidBodyRef]);

    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const globalMatrix = GlobalTransform.matrix.getBatch(eid);
      setMatrixTranslate(
        globalMatrix,
        RigidBodyState.position.get(eid, 0),
        RigidBodyState.position.get(eid, 1),
      );
      setMatrixRotateZ(globalMatrix, RigidBodyState.rotation[eid]);
    }
  };
}
