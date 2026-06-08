import { mat4 } from 'gl-matrix';
import { Not, query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

/**
 * Render-only attachments: entities that follow a physics-driven parent without
 * a rigid body of their own.
 *
 * The generic TransformSystem propagates parent -> child globals, but it runs
 * BEFORE the physics step, so a bodiless child would lag a frame behind and get
 * its parent's pre-sync transform. This system runs right after the rigid-body
 * sync and recomputes `global = parentGlobal * local` for [Parent, transforms,
 * no RigidBodyRef] entities — one hierarchy level, like the Children graph itself.
 */
export function createAttachedTransformSystem({ world } = GameDI) {
    const { Parent, LocalTransform, GlobalTransform, RigidBodyRef } = getGameComponents(world);

    return function updateAttachedTransforms() {
        const entities = query(world, [Parent, LocalTransform, GlobalTransform, Not(RigidBodyRef)]);

        for (let i = 0; i < entities.length; i++) {
            const eid = entities[i];
            const parentGlobal = GlobalTransform.matrix.getBatch(Parent.id[eid]);
            const local = LocalTransform.matrix.getBatch(eid);
            const global = GlobalTransform.matrix.getBatch(eid);
            mat4.multiply(global, parentGlobal, local);
        }
    };
}
