import { hasComponent, removeEntity } from 'bitecs';
import { removeRigidShape } from '../../Physical/createRigid.ts';
import { Worlds } from '../../DI/Worlds.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';

export function scheduleRemoveEntity(physEid: number, recursive = true, { physicsWorld } = Worlds) {
    const { Destroy } = getPhysicsWorldComponents(physicsWorld);
    Destroy.addComponent(physicsWorld, physEid, recursive);
}

// Reaps a single physics atom + its render mirror, plus unlinks Bridge.
export function typicalRemoveEntity(physEid: number, disconnect = true, { physicsWorld, renderWorld, physicalWorld } = Worlds) {
    const P = getPhysicsWorldComponents(physicsWorld);
    const R = getRenderWorldComponents(renderWorld);

    const renderEid = BridgeDI.getRenderOf(physEid);

    // Free the Rapier body (+ colliders + joints) and unregister the physical body id.
    const pid = P.RigidBodyRef.id[physEid];
    if (pid !== 0) {
        removeRigidShape(physicsWorld, physicalWorld, physEid);
        BridgeDI.unregisterPhysicalId(pid);
        P.RigidBodyRef.clear(physEid);
    }

    if (renderEid !== 0) {
        if (disconnect
            && hasComponent(renderWorld, renderEid, R.Parent)
            && hasComponent(renderWorld, R.Parent.id[renderEid], R.Children)) {
            R.Children.removeChild(R.Parent.id[renderEid], renderEid);
        }
        removeEntity(renderWorld, renderEid);
    }

    BridgeDI.unlink('mirror', physEid);
    removeEntity(physicsWorld, physEid);
}

// Walks the RenderWorld Children graph (ownership = render hierarchy in Step 1), reaping
// each child's atom+mirror; render-only children (slots/fx) are removed directly.
export function recursiveTypicalRemoveEntity(physEid: number, isRoot = true, { renderWorld } = Worlds) {
    const R = getRenderWorldComponents(renderWorld);

    const renderEid = BridgeDI.getRenderOf(physEid);
    if (renderEid !== 0) {
        removeRenderSubtree(renderWorld, renderEid, R);
    }

    typicalRemoveEntity(physEid, isRoot);
}

function removeRenderSubtree(
    renderWorld: RenderGameWorld,
    renderEid: number,
    R: ReturnType<typeof getRenderWorldComponents>,
) {
    if (!hasComponent(renderWorld, renderEid, R.Children)) return;

    const count = R.Children.entitiesCount[renderEid];
    for (let i = 0; i < count; i++) {
        const childRenderEid = R.Children.entitiesIds.get(renderEid, i);
        if (childRenderEid === 0) continue;
        const childPhysEid = BridgeDI.getPhysicsOf(childRenderEid);
        if (childPhysEid !== 0) {
            // child is an atom+mirror pair: reap recursively without disconnecting
            // (parent subtree is going away wholesale).
            recursiveTypicalRemoveEntity(childPhysEid, false);
        } else {
            // render-only child (slot / fx / exhaust): reap its subtree then itself.
            removeRenderSubtree(renderWorld, childRenderEid, R);
            removeEntity(renderWorld, childRenderEid);
        }
    }
    R.Children.entitiesCount[renderEid] = 0;
}
