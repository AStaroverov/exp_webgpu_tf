import { addComponent, observe, onRemove, World } from "bitecs";
import { defineComponent } from "../../../../renderer3d_2/src/ECS/utils.ts";
import {
  despawnBody,
  encodeOp,
  toSpawnOp,
  type BodySpec,
} from "../../Physics/opChannel.ts";

// Per-frame physics snapshot in 3D. The single biggest 2D→3D delta from the
// unknown package: position is [x,y,z], rotation is a QUATERNION [x,y,z,w]
// (2D stored a scalar angle), and linvel/angvel are [x,y,z].
//
// BRIDGE STORAGE (plan §3/§5.2): position + rotation are DOUBLE-BUFFERED on the
// shared DATA SAB (one view per bank). The writer fills the back bank then bumps
// SEQ to publish; readers address the last-published bank — so a 7-float pose is
// never read half-written. linvel/angvel are single-buffered shared columns
// (debug-only; tearing tolerated).
//
// The shared SAB is reached through `ctx.sab` (integrated into defineComponent like
// ctx.table): `ctx.sab.banks(name)` returns the per-bank views bound at the registry's
// byte offsets, so both threads point at byte-identical memory and this component does
// no offset math and takes no SAB constructor argument. There is no single-thread
// fallback (plan §2) — a world without a shared SAB throws on ctx.sab.
export const createRigidBodyStateComponent = defineComponent((RigidBodyState, ctx) => {
  const sab = ctx.sab;
  const positionBanks = sab.banks("RigidBodyState.position"); // banks:2
  const rotationBanks = sab.banks("RigidBodyState.rotation"); // banks:2
  const linvel = sab.banks("RigidBodyState.linvel")[0]; // banks:1
  const angvel = sab.banks("RigidBodyState.angvel")[0]; // banks:1

  const writePosition = () => positionBanks[sab.writeBank()];
  const writeRotation = () => rotationBanks[sab.writeBank()];

  // Despawn is ECS-driven (no public method): removing the entity (or this component) on
  // the PRODUCER world emits a DESPAWN_BODY op for the worker. The consumer (worker) also
  // removeEntity()s while applying ops, so it must NOT re-emit — gate on isProducer (else
  // an infinite spawn/despawn echo between the threads).
  observe(ctx.world, onRemove(RigidBodyState), (eid: number) => {
    if (sab.isProducer) sab.pushOp((payload, slot) => encodeOp(despawnBody(eid), payload, slot));
  });

  // Read accessors for position/rotation. NOTE: these MUST be method objects,
  // NOT getters — defineComponent does Object.assign(ref, create(...)), which
  // copies a getter's VALUE once (snapshotting a stale bank). A small wrapper
  // with get/getBatch resolves the active READ bank on every call, and keeps the
  // existing `RigidBodyState.position.get(eid, i)` call sites unchanged.
  const readAccessor = (banks: typeof positionBanks) => ({
    get: (eid: number, index: number) => banks[sab.readBank()].get(eid, index),
    getBatch: (eid: number) => banks[sab.readBank()].getBatch(eid),
  });

  return {
    // Read-bank accessors for readers (applyRigidBodyToTransform): bank resolved
    // per call so a publish() flip is honored immediately.
    position: readAccessor(positionBanks),
    rotation: readAccessor(rotationBanks),
    linvel,
    angvel,
    // Giving an entity RigidBodyState IS the spawn command (the public ECS surface — no
    // postOps method call). The BodySpec is REQUIRED: it seeds BOTH pose banks to the
    // spawn pose (so the shape renders there immediately, before the worker's first
    // publish) and — on the PRODUCER — emits a SPAWN_BODY op for the worker to materialize
    // the Rapier body. The worker also calls this (passing the drained op as the spec):
    // it re-seeds the shared banks (same values) and, being a CONSUMER, does NOT re-emit.
    addComponent(world: World, eid: number, spec: BodySpec) {
      addComponent(world, eid, RigidBodyState);
      // Identity quaternion (w = 1) + the body CENTER, written to both banks.
      for (const p of positionBanks) {
        p.set(eid, 0, spec.position.x).set(eid, 1, spec.position.y).set(eid, 2, spec.position.z);
      }
      for (const r of rotationBanks) {
        r.getBatch(eid).fill(0);
        r.set(eid, 3, 1);
      }
      linvel.getBatch(eid).fill(0);
      angvel.getBatch(eid).fill(0);
      if (sab.isProducer) {
        sab.pushOp((payload, slot) => encodeOp(toSpawnOp(eid, spec), payload, slot));
      }
    },
    update(
      eid: number,
      px: number,
      py: number,
      pz: number,
      qx: number,
      qy: number,
      qz: number,
      qw: number,
      lx: number,
      ly: number,
      lz: number,
      ax: number,
      ay: number,
      az: number,
    ) {
      // Writer fills the BACK (write) bank; publish() flips it to readers.
      const pos = writePosition();
      const rot = writeRotation();
      pos.set(eid, 0, px).set(eid, 1, py).set(eid, 2, pz);
      rot.set(eid, 0, qx).set(eid, 1, qy).set(eid, 2, qz).set(eid, 3, qw);
      linvel.set(eid, 0, lx).set(eid, 1, ly).set(eid, 2, lz);
      angvel.set(eid, 0, ax).set(eid, 1, ay).set(eid, 2, az);
    },
  };
});
