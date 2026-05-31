import { query } from 'bitecs';
import {
    LocalTransform,
    setMatrixTranslate,
    setMatrixRotateZ,
} from '../../../../../renderer/src/ECS/Components/Transform.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';
import { Worlds } from '../../DI/Worlds.ts';

// For each AWAKE physics atom: translate physicsEid -> renderEid via Bridge, copy
// RigidBodyState(x,y,rot) to the mirror's LocalTransform. Runs in the RENDER tick,
// BEFORE TransformSystem (which composes Local -> Global and propagates to children).
export function createMirrorSyncSystem({ physicsWorld, physicalWorld } = Worlds) {
    const { RigidBodyRef, RigidBodyState } = getPhysicsWorldComponents(physicsWorld);

    return function () {
        const atoms = query(physicsWorld, [RigidBodyRef, RigidBodyState]);
        for (let i = 0; i < atoms.length; i++) {
            const physEid = atoms[i];
            const body = physicalWorld.getRigidBody(RigidBodyRef.id[physEid]);
            if (body && body.isSleeping()) continue;
            const renderEid = BridgeDI.getRenderOf(physEid);
            if (renderEid === 0) continue;
            const m = LocalTransform.matrix.getBatch(renderEid);
            setMatrixTranslate(m, RigidBodyState.position.get(physEid, 0), RigidBodyState.position.get(physEid, 1));
            setMatrixRotateZ(m, RigidBodyState.rotation[physEid]);
        }
    };
}
