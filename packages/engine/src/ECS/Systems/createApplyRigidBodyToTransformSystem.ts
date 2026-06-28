import { query } from "bitecs";
import { mat4, quat, vec3 } from "gl-matrix";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents } from "../createEngineWorld.ts";

// state → LocalTransform.matrix. The load-bearing matrix write.
//
// Center-origin: the render transform's translation IS the body center, and the SDF is
// symmetric about local z=0, so we write Rapier's body CENTROID straight into the matrix
// with no Z offset (and no Shape read). render center == physics center exactly.
//
// We build the full 3D rotation via mat4.fromRotationTranslation even though the
// SDF pass only honors yaw (and spheres are rotation-invariant) — it costs the
// same and is future-proof for a shader upgrade (see plan §1/§8).
export function createApplyRigidBodyToTransformSystem(world: EngineWorld): () => void {
  // RigidBodyRef is WORKER-LOCAL at Step 3 (main never uses pid), so it is NOT in the
  // query — main matches on RigidBodyState (added by RigidShapes' main half).
  const { LocalTransform, RigidBodyState } = getEngineComponents(world);

  const _q = quat.create();
  const _t = vec3.create();
  let frameLog = 0;

  return function applyRigidBodyToLocalTransform() {
    const entities = query(world, [LocalTransform, RigidBodyState]);
    if (frameLog < 10) {
      const e0 = entities[0];
      console.log(
        "[main] apply frame#", frameLog,
        "entities", entities.length,
        e0 !== undefined
          ? `eid0=${e0} pose=(${RigidBodyState.position.get(e0, 0).toFixed(2)},${RigidBodyState.position.get(e0, 1).toFixed(2)},${RigidBodyState.position.get(e0, 2).toFixed(2)})`
          : "(no entities)",
      );
      frameLog++;
    }
    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];

      quat.set(
        _q,
        RigidBodyState.rotation.get(eid, 0),
        RigidBodyState.rotation.get(eid, 1),
        RigidBodyState.rotation.get(eid, 2),
        RigidBodyState.rotation.get(eid, 3),
      );
      vec3.set(
        _t,
        RigidBodyState.position.get(eid, 0),
        RigidBodyState.position.get(eid, 1),
        RigidBodyState.position.get(eid, 2), // center-origin: position IS the render center
      );

      const m = LocalTransform.matrix.getBatch(eid) as unknown as mat4;
      mat4.fromRotationTranslation(m, _q, _t);
    }
  };
}
