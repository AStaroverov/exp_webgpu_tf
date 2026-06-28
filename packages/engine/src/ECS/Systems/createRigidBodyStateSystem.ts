import { query } from "bitecs";
import type { EngineWorld, PhysicsWorkerWorld } from "../createEngineWorld.ts";
import { getEngineComponents } from "../createEngineWorld.ts";
import type { PhysicalWorld } from "../../Physics/initPhysicalWorld.ts";

export function createRigidBodyStateSystem(
  world: EngineWorld | PhysicsWorkerWorld,
  physicalWorld: PhysicalWorld,
): () => void {
  const { RigidBodyRef, RigidBodyState } = getEngineComponents(world);

  return function syncRigidBodyState() {
    const entities = query(world, [RigidBodyRef, RigidBodyState]);
    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const pid = RigidBodyRef.id[eid];
      const body = physicalWorld.getRigidBody(pid);
      const p = body.translation(); // {x,y,z} — physics CENTER
      const q = body.rotation(); // {x,y,z,w} quaternion
      const lv = body.linvel(); // {x,y,z}
      const av = body.angvel(); // {x,y,z}
      RigidBodyState.update(
        eid,
        p.x,
        p.y,
        p.z,
        q.x,
        q.y,
        q.z,
        q.w,
        lv.x,
        lv.y,
        lv.z,
        av.x,
        av.y,
        av.z,
      );
    }
  };
}
