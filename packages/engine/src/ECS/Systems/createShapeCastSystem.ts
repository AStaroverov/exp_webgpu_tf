import { query } from "bitecs";
import { vec3 } from "gl-matrix";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents, getEngineSab } from "../createEngineWorld.ts";
import { castShape, encodeOp } from "../../Physics/opChannel.ts";

// PRODUCER half of the shape-cast query: [ShapeCaster, GlobalTransform] → one
// CAST_SHAPE op per entity per frame. The local segment is lifted into world space
// with the entity's GlobalTransform, so this runs AFTER the transform system and
// knows nothing about what moved the entity (animation, physics, teleport).
export function createShapeCastSystem(world: EngineWorld): () => void {
  const { ShapeCaster, GlobalTransform } = getEngineComponents(world);
  const sab = getEngineSab(world);

  let nextQueryId = 0;
  const local = new Float64Array(3);
  const from = { x: 0, y: 0, z: 0 };
  const to = { x: 0, y: 0, z: 0 };
  const scratch = vec3.create();

  return function requestShapeCasts() {
    if (!sab.isProducer) return;
    const entities = query(world, [ShapeCaster, GlobalTransform]);
    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const matrix = GlobalTransform.matrix.getBatch(eid);

      ShapeCaster.localFrom.getBatch(eid, local);
      vec3.transformMat4(scratch, local as unknown as vec3, matrix);
      from.x = scratch[0];
      from.y = scratch[1];
      from.z = scratch[2];

      ShapeCaster.localTo.getBatch(eid, local);
      vec3.transformMat4(scratch, local as unknown as vec3, matrix);
      to.x = scratch[0];
      to.y = scratch[1];
      to.z = scratch[2];

      const queryId = ++nextQueryId;
      const radius = ShapeCaster.radius.get(eid);
      const excludeEid = ShapeCaster.excludeEid.get(eid);
      ShapeCaster.queryId.set(eid, queryId);
      sab.pushOp((payload, slot) =>
        encodeOp(castShape(queryId, eid, excludeEid, from, to, radius), payload, slot),
      );
    }
  };
}
