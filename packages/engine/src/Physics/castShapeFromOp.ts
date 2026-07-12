import { Capsule } from "@dimforge/rapier3d-simd";
import type { PhysicalWorld } from "./initPhysicalWorld.ts";
import type { CastShapeOp } from "./opChannel.ts";

// WORKER half of the CAST_SHAPE query: intersect a capsule spanning the op's from→to
// segment (thickness = radius) against every collider in the world, excluding the
// source entity's own body, and emit one callback per intersected body. The caller
// (physics.worker) routes each emit into the HITS ring.
//
// Rapier's Capsule is aligned with its local +Y axis, so the segment becomes
// midpoint + the shortest-arc rotation taking +Y onto the segment direction. A
// zero-length segment degenerates to halfHeight 0 — a ball — which is exactly the
// "just probe this point" semantics.

// One scratch capsule, mutated per query (its fields are plain JS data that Rapier
// converts to raw on each call) — no per-query allocation.
const capsule = new Capsule(0, 1);
const pos = { x: 0, y: 0, z: 0 };
const rot = { x: 0, y: 0, z: 0, w: 1 };

export function castShapeFromOp(
  physicalWorld: PhysicalWorld,
  op: CastShapeOp,
  excludePid: number,
  emit: (hitPid: number) => void,
): void {
  const dx = op.toX - op.fromX;
  const dy = op.toY - op.fromY;
  const dz = op.toZ - op.fromZ;
  const len = Math.hypot(dx, dy, dz);

  capsule.halfHeight = len / 2;
  capsule.radius = op.radius;
  pos.x = (op.fromX + op.toX) / 2;
  pos.y = (op.fromY + op.toY) / 2;
  pos.z = (op.fromZ + op.toZ) / 2;
  setRotationFromYAxis(dx, dy, dz, len);

  const excludeBody = excludePid !== 0 ? physicalWorld.getRigidBody(excludePid) : undefined;

  physicalWorld.intersectionsWithShape(
    pos,
    rot,
    capsule,
    (collider) => {
      const body = collider.parent();
      if (body) emit(body.handle);
      return true; // keep enumerating — the query returns ALL intersections
    },
    undefined,
    undefined,
    undefined,
    excludeBody ?? undefined,
  );
}

// Shortest-arc quaternion taking the capsule's local +Y onto the segment direction:
// q = normalize(cross(Y, d), w = 1 + dot(Y, d)). For d ≈ −Y the arc is degenerate
// (w → 0, cross → 0); any axis ⊥ Y works, so use a half-turn about X.
function setRotationFromYAxis(dx: number, dy: number, dz: number, len: number): void {
  if (len < 1e-9) {
    rot.x = 0;
    rot.y = 0;
    rot.z = 0;
    rot.w = 1;
    return;
  }
  const nx = dx / len;
  const ny = dy / len;
  const nz = dz / len;
  const w = 1 + ny;
  if (w < 1e-9) {
    rot.x = 1;
    rot.y = 0;
    rot.z = 0;
    rot.w = 0;
    return;
  }
  const qx = nz;
  const qz = -nx;
  const inv = 1 / Math.hypot(qx, qz, w);
  rot.x = qx * inv;
  rot.y = 0;
  rot.z = qz * inv;
  rot.w = w * inv;
}
