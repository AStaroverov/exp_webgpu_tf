import { query } from 'bitecs';
import {
    LocalTransform,
    setMatrixTranslate,
    setMatrixRotateZ,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../createRenderWorld.ts';
import { getPhysicsOf } from '../refs.ts';
import { Worlds } from '../../DI/Worlds.ts';

// Render-centric: for each render mirror that has a physics body (render PhysicsRef),
// pull its body's RigidBodyState(x,y,rot) into the mirror's LocalTransform. Sleeping
// bodies are skipped. Runs in the RENDER tick, BEFORE TransformSystem (which composes
// Local -> Global and propagates to children). Single downward arrow render -> physics.
export function createMirrorSyncSystem({ physicsWorld, renderWorld, physicalWorld } = Worlds) {
    const { RigidBodyRef, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);
    const { PhysicsRef } = getRenderWorldComponents(renderWorld);

    return function () {
        const renders = query(renderWorld, [PhysicsRef]);
        for (let i = 0; i < renders.length; i++) {
            const renderEid = renders[i];
            const physEid = getPhysicsOf(renderEid);
            if (physEid === 0) continue;
            const body = physicalWorld.getRigidBody(RigidBodyRef.id[physEid]);
            if (body && body.isSleeping()) continue;
            const m = LocalTransform.matrix.getBatch(renderEid);
            setMatrixTranslate(m, RigidBodyState.position.get(physEid, 0), RigidBodyState.position.get(physEid, 1));
            setMatrixRotateZ(m, RigidBodyState.rotation[physEid]);
        }
    };
}
