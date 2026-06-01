import { hasComponent, query, removeEntity } from 'bitecs';
import { removeRigidShape } from '../../Physical/createRigid.ts';
import { Worlds } from '../../DI/Worlds.ts';
import {
    clearNodePhysics,
    clearNodeRender,
    clearNodeSlots,
    clearOccupant,
    clearSoundOwner,
    getNodeByPhysics,
    getNodeChildren,
    getNodePhysics,
    getNodeRender,
    getNodeSlots,
    getSoundsOfOwner,
} from '../refs.ts';
import { physicsByBody } from '../../DI/physicsByBody.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../createRenderWorld.ts';
import { getSoundWorldComponents } from '../createSoundWorld.ts';

export function scheduleRemoveEntity(physEid: number, recursive = true, { physicsWorld } = Worlds) {
    const { Destroy } = getPhysicsWorldComponents(physicsWorld);
    Destroy.addComponent(physicsWorld, physEid, recursive);
}

// Reaps a single physics atom + its render mirror (and its brain node, if any). Used
// for node-less leaf atoms (bullets, torn-off parts, fx) and as the per-node step of
// the recursive Brain-tree teardown below. `disconnect` detaches this render from its
// render parent's Children (skip when the whole parent subtree is being reaped).
export function typicalRemoveEntity(physEid: number, disconnect = true, worlds = Worlds) {
    const { physicsWorld, renderWorld, soundWorld, physicalWorld } = worlds;
    const P = getPhysicsWorldComponents(physicsWorld);
    const R = getRenderWorldComponents(renderWorld);

    // The render that mirrors this atom: a brain node points at it downward (node ->
    // render); a node-less leaf carries it on no node, so we fall back to the slot/none.
    const node = getNodeByPhysics(physEid, worlds);
    const renderEid = node !== 0 ? getNodeRender(node, worlds) : findRenderOfLeaf(physEid, worlds);

    // Reap the SoundWorld sounds this atom owned (engine-move loop, etc.) — sounds live
    // in SoundWorld with no render entity, so they are NOT reaped by any render walk.
    const ownedSounds = Array.from(getSoundsOfOwner(physEid, worlds));
    if (ownedSounds.length > 0) {
        const { Sound } = getSoundWorldComponents(soundWorld);
        for (const soundEid of ownedSounds) {
            clearSoundOwner(soundEid, worlds);
            Sound.removeComponent(soundWorld, soundEid);
            removeEntity(soundWorld, soundEid);
        }
    }

    // Reap the SlotWorld slots this node owned (slots are SlotWorld entities, not render
    // children, so a render walk never reaps them). Matches the pre-split carrier-slot
    // walk; the node owns exactly the slots its carrier atom used to.
    if (node !== 0) {
        for (const slotEid of getNodeSlots(node, worlds)) {
            clearOccupant(slotEid, worlds);
            removeEntity(Worlds.slotWorld, slotEid);
        }
        clearNodeSlots(node, worlds);
    }

    // Free the Rapier body (+ colliders + joints) and unregister the physical body id.
    const pid = P.RigidBodyRef.id[physEid];
    if (pid !== 0) {
        removeRigidShape(physicsWorld, physicalWorld, physEid);
        physicsByBody.delete(pid);
        P.RigidBodyRef.clear(physEid);
    }

    // Reap the render mirror: drop the render -> physics link, detach from its render
    // parent (when standalone), then the render entity. Render-only children (exhaust
    // pipes/fx) are reaped by removeNodeTree before the node tree is descended.
    if (renderEid !== 0) {
        R.PhysicsRef.clear(renderWorld, renderEid);
        if (disconnect
            && hasComponent(renderWorld, renderEid, R.Parent)
            && hasComponent(renderWorld, R.Parent.id[renderEid], R.Children)) {
            R.Children.removeChild(R.Parent.id[renderEid], renderEid);
        }
        removeEntity(renderWorld, renderEid);
    }

    // Reap the brain node (its render/physics refs and the node entity).
    if (node !== 0) {
        clearNodeRender(node, worlds);
        clearNodePhysics(node, worlds);
        removeEntity(Worlds.brainWorld, node);
    }

    removeEntity(physicsWorld, physEid);
}

// Walks the Brain node tree (a node + its Brain Children, descending) and reaps each
// node together with its presentation (render + physics atom), Rapier body, sounds and
// slots. This is the post-split replacement for the old OwnerRef + OwnedGraph + render
// -Children walk; it ALSO reaps the track/wheel/gun brain NODES that the render walk
// left behind (the Phase-1 leak), since those nodes are now first-class tree members.
export function recursiveTypicalRemoveEntity(physEid: number, isRoot = true, worlds = Worlds) {
    const node = getNodeByPhysics(physEid, worlds);
    if (node === 0) {
        // Node-less leaf (bullet / torn-off part / fx): reap it directly.
        typicalRemoveEntity(physEid, isRoot, worlds);
        return;
    }
    removeNodeTree(node, isRoot, worlds);
}

function removeNodeTree(node: number, isRoot: boolean, worlds: typeof Worlds) {
    // Reap this node's render-only children (exhaust pipes / fx) FIRST, while its
    // atom-backed child renders (turret/track/wheel) are still alive so they are
    // correctly skipped (they get reaped via their own nodes below).
    const renderEid = getNodeRender(node, worlds);
    if (renderEid !== 0) {
        const R = getRenderWorldComponents(worlds.renderWorld);
        removeRenderOnlyChildren(worlds.renderWorld, renderEid, R);
    }

    // Descend into the child nodes (snapshot before mutation), depth-first.
    const children = getNodeChildren(node, worlds);
    for (let i = 0; i < children.length; i++) {
        removeNodeTree(children[i], false, worlds);
    }

    const physEid = getNodePhysics(node, worlds);
    if (physEid !== 0) {
        // Reaps the node's physics atom, render, sounds, slots and the node entity.
        // Only the root keeps its render parent detach (children go away wholesale).
        typicalRemoveEntity(physEid, isRoot, worlds);
    } else {
        // Defensive: a node with no physics presentation — reap its render + node.
        if (renderEid !== 0) removeEntity(worlds.renderWorld, renderEid);
        clearNodeRender(node, worlds);
        clearNodeSlots(node, worlds);
        removeEntity(worlds.brainWorld, node);
    }
}

// Reaps the render-only (non-atom-backed) descendants of a render entity — exhaust
// pipes, fx — which the brain-tree walk does not own. Atom-backed render children are
// reaped via their own brain nodes, so they are skipped here.
function removeRenderOnlyChildren(
    renderWorld: RenderGameWorld,
    renderEid: number,
    R: ReturnType<typeof getRenderWorldComponents>,
) {
    if (!hasComponent(renderWorld, renderEid, R.Children)) return;

    const count = R.Children.entitiesCount[renderEid];
    for (let i = 0; i < count; i++) {
        const childRenderEid = R.Children.entitiesIds.get(renderEid, i);
        if (childRenderEid === 0) continue;
        // Atom-backed render children (turret/track/wheel mirrors) are reaped via their
        // own brain nodes; skip them here and reap only render-only leaves.
        if (hasComponent(renderWorld, childRenderEid, R.PhysicsRef.ref)) continue;
        removeRenderOnlyChildren(renderWorld, childRenderEid, R);
        removeEntity(renderWorld, childRenderEid);
    }
    R.Children.entitiesCount[renderEid] = 0;
}

// Resolves the render of a node-less leaf atom (no brain node, no node->render ref):
// scan render mirrors for the one whose PhysicsRef points at this atom. Cold — only
// for bullets / torn-off parts / fx at their own teardown. Keeps the single downward
// arrow render -> physics, with no reverse ref on the atom.
function findRenderOfLeaf(physEid: number, { renderWorld }: typeof Worlds): number {
    const { PhysicsRef } = getRenderWorldComponents(renderWorld);
    for (const renderEid of query(renderWorld, [PhysicsRef.ref])) {
        if (PhysicsRef.id[renderEid] === physEid) return renderEid;
    }
    return 0;
}
