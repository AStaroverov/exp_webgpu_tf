import { query } from "bitecs";
import type { EngineWorld, PhysicsWorkerWorld } from "../createEngineWorld.ts";
import { getEngineComponents } from "../createEngineWorld.ts";
import type { PhysicalWorld } from "../../Physics/initPhysicalWorld.ts";

// READ direction: Rapier body → RigidBodyState snapshot. The high-level
// RigidBody accessors (translation/rotation/linvel/angvel) return plain GC'd JS
// objects, so there are no WASM handles to .free() here (unlike the unknown
// package's raw-bodies path).
// Runs in the PHYSICS thread (worker at Step 3). Accepts either world flavor: the
// main engine world (render + engine) or the worker world (engine-only) — it reads
// only the physics-bridge components, present on both.
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
      // NOTE: we intentionally do NOT skip sleeping bodies. With the double-buffered
      // pose (plan §5.2), skipping would leave a sleeping body's row STALE in the
      // back bank after a bank flip → it would teleport to an old pose on read.
      // Writing the (constant) sleeping pose into the back bank every step keeps both
      // banks complete and is negligible vs. the step itself (Spike 5 resolution (b)).
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
