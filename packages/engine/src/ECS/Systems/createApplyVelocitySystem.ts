import { query } from "bitecs";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents, getEngineSab } from "../createEngineWorld.ts";
import { encodeOp, setVelocity } from "../../Physics/opChannel.ts";

// PRODUCER half of the velocity-drive: Velocity component → SET_VELOCITY op. Mirror of
// RigidBodyState's spawn/despawn op production — the system is the only place that writes
// the ops line for velocity, and the worker is the only place that touches Rapier. Pushes
// one op per body each frame (a body with no controller keeps writing its last velocity,
// which is exactly the persistent-velocity semantics Rapier already has).
export function createApplyVelocitySystem(world: EngineWorld): () => void {
  const { RigidBodyState, Velocity } = getEngineComponents(world);
  const sab = getEngineSab(world);

  return function applyVelocity() {
    if (!sab.isProducer) return;
    const entities = query(world, [RigidBodyState, Velocity]);
    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const vx = Velocity.x.get(eid);
      const vy = Velocity.y.get(eid);
      const vz = Velocity.z.get(eid);
      sab.pushOp((payload, slot) => encodeOp(setVelocity(eid, vx, vy, vz), payload, slot));
    }
  };
}
