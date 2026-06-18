import { mat4 } from "gl-matrix";
import { query } from "bitecs";
import { GlobalTransform, LocalTransform } from "renderer/src/ECS/Components/Transform.ts";
import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";

/**
 * Positions compound armor parts — colliders living on a parent body, with no
 * rigid body of their own. Their world transform is `ownerGlobal * translate(
 * anchor)`: the anchor is a fixed offset in the owner's local frame, so the
 * owner's rotation is already baked into `ownerGlobal`. Runs right after the
 * rigid-body sync (same slot as `updateAttachedTransforms`), so the owner's
 * global is fresh this frame. One hierarchy level, monomorphic loop.
 *
 * The part's z (height for the RC lighting / shadows) is NOT physics-driven —
 * it lives in the part's own LocalTransform (set at spawn from the slot config)
 * and is copied back after the planar compose, which would otherwise zero it.
 */
export function createCompoundPartTransformSystem({ world } = GameDI) {
  const { CompoundPart } = getGameComponents(world);
  const local = mat4.create();

  return function updateCompoundPartTransforms() {
    const entities = query(world, [CompoundPart, GlobalTransform]);

    for (let i = 0; i < entities.length; i++) {
      const eid = entities[i];
      const ownerGlobal = GlobalTransform.matrix.getBatch(CompoundPart.ownerEid.get(eid));
      const global = GlobalTransform.matrix.getBatch(eid);
      local[12] = CompoundPart.anchorX.get(eid);
      local[13] = CompoundPart.anchorY.get(eid);
      mat4.multiply(global, ownerGlobal, local);
      // Restore the configured height (z) — the planar compose above zeroes it,
      // but the lighting pass reads it to cast the part's shadow.
      global[14] = LocalTransform.matrix.get(eid, 14);
    }
  };
}
