import { query } from "bitecs";
import { mat4, quat, vec3 } from "gl-matrix";
import type { EngineWorld } from "../createEngineWorld.ts";
import { getEngineComponents } from "../createEngineWorld.ts";

// state → LocalTransform.matrix. The load-bearing matrix write.
//
// Z-origin is the BOTTOM of the shape (sdf.shader.ts: transform col3.z = baseZ;
// the shader lifts center.z = baseZ + Height/2). Rapier's body translation is the
// CENTROID, so we write baseZ = centerZ − Height/2. Because the entity factory
// sets Height = the body's full Z extent, the shader recovers
// center.z = baseZ + Height/2 = centerZ exactly.
//
// We build the full 3D rotation via mat4.fromRotationTranslation even though the
// SDF pass only honors yaw (and spheres are rotation-invariant) — it costs the
// same and is future-proof for a shader upgrade (see plan §1/§8).
export function createApplyRigidBodyToTransformSystem(world: EngineWorld): () => void {
  // RigidBodyRef is WORKER-LOCAL at Step 3 (main never uses pid), so it is NOT in the
  // query — main matches on RigidBodyState (added by RigidShapes' main half) + Height.
  const { LocalTransform, RigidBodyState, Height } = getEngineComponents(world);

  const _q = quat.create();
  const _t = vec3.create();
  let frameLog = 0;

  return function applyRigidBodyToLocalTransform() {
    const entities = query(world, [LocalTransform, RigidBodyState, Height]);
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
      const hz = Height.value[eid] * 0.5; // half vertical extent

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
        RigidBodyState.position.get(eid, 2) - hz, // baseZ = centerZ − hz
      );

      const m = LocalTransform.matrix.getBatch(eid) as unknown as mat4;
      mat4.fromRotationTranslation(m, _q, _t);
    }
  };
}
