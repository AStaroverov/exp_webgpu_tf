import { query } from "bitecs";
import { mat4, quat, vec3 } from "gl-matrix";
import type { EngineWorld } from "../createEngineWorld.ts";
import { createEntityId, getEngineComponents } from "../createEngineWorld.ts";
import { removeEntityTree } from "../hierarchy.ts";
import { ColliderKind } from "../Components/RigidBodyState.ts";
import {
  createCircle,
  createRectangle,
  createSphere,
} from "../../../../renderer/src/ECS/Entities/Shapes.ts";
import type { TColor } from "../../../../renderer/src/ECS/Components/Common.ts";

// Classic-engine collider visualization: while enabled, every physics volume gets a
// translucent ghost shape — bodies (RigidBodyState) as their sphere/box at the pose
// the worker publishes, shape-cast queries (ShapeCaster) as a cylinder along the
// blade's world segment. Ghosts are ordinary render entities parented to the scene
// root, created when a source appears and torn down when it vanishes (or when the
// toggle goes off), so the system leaves zero residue while disabled.
export const ColliderDebug = { enabled: false };

const BODY_COLOR: TColor = [0.25, 0.9, 0.4, 0.35];
const CASTER_COLOR: TColor = [1.0, 0.55, 0.15, 0.4];

export function createColliderDebugSystem(world: EngineWorld, sceneRoot: number): () => void {
  const { Children, LocalTransform, GlobalTransform, RigidBodyState, ShapeCaster } =
    getEngineComponents(world);

  // source eid → ghost eid. Ghost lifetime is mark-and-sweep against the queries:
  // debug-only path, clarity over hot-loop discipline.
  const ghosts = new Map<number, number>();
  const seen = new Set<number>();

  const _q = quat.create();
  const _t = vec3.create();
  const _s = vec3.create();
  const _from = vec3.create();
  const _to = vec3.create();
  const _dir = vec3.create();
  const _local = new Float64Array(3);
  const Z_AXIS: vec3 = [0, 0, 1];

  function ensureGhost(sourceEid: number, create: () => number): number {
    let ghost = ghosts.get(sourceEid);
    if (ghost === undefined) {
      ghost = create();
      Children.addChild(sceneRoot, ghost);
      ghosts.set(sourceEid, ghost);
    }
    return ghost;
  }

  function dropGhost(sourceEid: number, ghost: number): void {
    Children.removeChild(sceneRoot, ghost);
    removeEntityTree(world, ghost);
    ghosts.delete(sourceEid);
  }

  function sweep(): void {
    for (const [sourceEid, ghost] of ghosts) {
      if (!seen.has(sourceEid)) dropGhost(sourceEid, ghost);
    }
  }

  function createBodyGhost(eid: number): number {
    const kind = RigidBodyState.getColliderKind(eid);
    if (kind === ColliderKind.sphere) {
      return createSphere(world, {
        x: 0,
        y: 0,
        z: 0,
        radius: RigidBodyState.getColliderDim(eid, 0),
        color: BODY_COLOR,
        eid: createEntityId(world),
      });
    }
    return createRectangle(world, {
      x: 0,
      y: 0,
      z: 0,
      width: RigidBodyState.getColliderDim(eid, 0) * 2,
      height: RigidBodyState.getColliderDim(eid, 1) * 2,
      depth: RigidBodyState.getColliderDim(eid, 2) * 2,
      color: BODY_COLOR,
      eid: createEntityId(world),
    });
  }

  // Unit cylinder (radius 1, height 1): the per-frame matrix scales it to the
  // segment's radius/length, so one Shape fits every caster and every frame.
  function createCasterGhost(): number {
    return createCircle(world, {
      x: 0,
      y: 0,
      z: 0,
      radius: 1,
      height: 1,
      color: CASTER_COLOR,
      eid: createEntityId(world),
    });
  }

  return function drawColliderDebug() {
    if (!ColliderDebug.enabled) {
      if (ghosts.size > 0) {
        seen.clear();
        sweep();
      }
      return;
    }
    seen.clear();

    const bodies = query(world, [RigidBodyState]);
    for (let i = 0; i < bodies.length; i++) {
      const eid = bodies[i];
      seen.add(eid);
      const ghost = ensureGhost(eid, () => createBodyGhost(eid));
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
        RigidBodyState.position.get(eid, 2),
      );
      mat4.fromRotationTranslation(
        LocalTransform.matrix.getBatch(ghost) as unknown as mat4,
        _q,
        _t,
      );
    }

    // Casters read the caster's GlobalTransform of the PREVIOUS frame (this system
    // runs before the transform system so its own ghosts get composed this frame) —
    // a one-frame-late ghost is fine for a debug overlay.
    const casters = query(world, [ShapeCaster, GlobalTransform]);
    for (let i = 0; i < casters.length; i++) {
      const eid = casters[i];
      // A caster entity that ALSO has a body keeps its body ghost; casters get
      // their own map key space via the negated eid (eids are never 0).
      seen.add(-eid);
      const ghost = ensureGhost(-eid, createCasterGhost);

      const matrix = GlobalTransform.matrix.getBatch(eid);
      ShapeCaster.localFrom.getBatch(eid, _local);
      vec3.transformMat4(_from, _local as unknown as vec3, matrix);
      ShapeCaster.localTo.getBatch(eid, _local);
      vec3.transformMat4(_to, _local as unknown as vec3, matrix);

      vec3.subtract(_dir, _to, _from);
      const len = vec3.length(_dir);
      if (len > 1e-9) vec3.scale(_dir, _dir, 1 / len);
      else vec3.copy(_dir, Z_AXIS);
      quat.rotationTo(_q, Z_AXIS, _dir);
      vec3.lerp(_t, _from, _to, 0.5);
      const radius = ShapeCaster.radius.get(eid);
      vec3.set(_s, radius, radius, Math.max(len, 1e-6));
      mat4.fromRotationTranslationScale(
        LocalTransform.matrix.getBatch(ghost) as unknown as mat4,
        _q,
        _t,
        _s,
      );
    }

    sweep();
  };
}
