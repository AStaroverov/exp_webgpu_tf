import { hasComponent, query } from 'bitecs';
import { Worlds } from '../DI/Worlds.ts';
import { getBrainWorldComponents } from './createBrainWorld.ts';
import { getPhysicsWorldComponents } from './createPhysicsWorld.ts';
import { getRenderWorldComponents } from './createRenderWorld.ts';
import { getSlotWorldComponents } from './createSlotWorld.ts';
import { getSoundWorldComponents } from './createSoundWorld.ts';

// ---- render -> physics (the single downward arrow between the two tiers) ----
export function getPhysicsOf(renderEid: number, { renderWorld } = Worlds): number { return getRenderWorldComponents(renderWorld).PhysicsRef.id[renderEid]; }

// ---- slots ----
// occupant: slot -> the occupying part's presentation (physics atom). Downward only;
// the part no longer keeps a reverse ref to its slot (occupancy lives on the slot).
export function attachOccupant(slotEid: number, partPhysEid: number, { slotWorld } = Worlds) {
    getSlotWorldComponents(slotWorld).OccupantRef.set(slotWorld, slotEid, partPhysEid);
}
export function clearOccupant(slotEid: number, { slotWorld } = Worlds) { getSlotWorldComponents(slotWorld).OccupantRef.clear(slotWorld, slotEid); }
export function getOccupantOf(slotEid: number, { slotWorld } = Worlds): number { return getSlotWorldComponents(slotWorld).OccupantRef.id[slotEid]; }

// ---- node refs (BrainWorld node -> downward presentation/slots; scheme A) ----
// A node carries NodeRenderRef XOR NodePhysicsRef (set by the constructor at spawn),
// plus NodeSlotsRef (1:N slot list). These resolve from a brain node eid, distinct
// from the render-side getPhysicsOf above (which resolves from a render eid).
export function setNodeRender(nodeEid: number, renderEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodeRenderRef.set(brainWorld, nodeEid, renderEid); }
export function clearNodeRender(nodeEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodeRenderRef.clear(brainWorld, nodeEid); }
export function setNodePhysics(nodeEid: number, physEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodePhysicsRef.set(brainWorld, nodeEid, physEid); }
export function clearNodePhysics(nodeEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodePhysicsRef.clear(brainWorld, nodeEid); }

// node -> render (0 if the node is not drawn).
export function getNodeRender(nodeEid: number, { brainWorld } = Worlds): number {
    const { NodeRenderRef } = getBrainWorldComponents(brainWorld);
    return hasComponent(brainWorld, nodeEid, NodeRenderRef.ref) ? NodeRenderRef.id[nodeEid] : 0;
}
// node -> physics: direct body if NodePhysicsRef, else through the render's PhysicsRef,
// else 0. The "has render / else physics" branch lives here (single resolver, per STRUCTURE.md).
export function getNodePhysics(nodeEid: number, worlds = Worlds): number {
    const { brainWorld } = worlds;
    const { NodePhysicsRef, NodeRenderRef } = getBrainWorldComponents(brainWorld);
    if (hasComponent(brainWorld, nodeEid, NodePhysicsRef.ref)) return NodePhysicsRef.id[nodeEid];
    if (hasComponent(brainWorld, nodeEid, NodeRenderRef.ref)) return getPhysicsOf(NodeRenderRef.id[nodeEid], worlds);
    return 0;
}

// node -> its slots (1:N). attach appends; getNodeSlots returns the list (empty if none).
export function attachSlotToNode(nodeEid: number, slotEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodeSlotsRef.attach(brainWorld, nodeEid, slotEid); }
export function clearNodeSlots(nodeEid: number, { brainWorld } = Worlds) { getBrainWorldComponents(brainWorld).NodeSlotsRef.clear(brainWorld, nodeEid); }
export function getNodeSlots(nodeEid: number, { brainWorld } = Worlds): number[] { return getBrainWorldComponents(brainWorld).NodeSlotsRef.get(nodeEid); }

// ---- Brain hierarchy (node -> node, within the tier) ----
// Links a child node under a parent node: writes Parent(child -> parent) and appends the
// child to the parent's Children list — the structural object tree per STRUCTURE.md.
export function linkBrainChild(parentNode: number, childNode: number, { brainWorld } = Worlds) {
    const { Parent, Children } = getBrainWorldComponents(brainWorld);
    Parent.addComponent(brainWorld, childNode, parentNode);
    if (!hasComponent(brainWorld, parentNode, Children)) {
        Children.addComponent(brainWorld, parentNode);
    }
    Children.addChildren(parentNode, childNode);
}
// Brain hierarchy reads (node -> node). getNodeParent returns 0 when the node has no
// Parent (e.g. a hull/root node). getNodeChildren returns the live child list (length =
// Children.entitiesCount), empty when the node has no Children component.
export function getNodeParent(nodeEid: number, { brainWorld } = Worlds): number {
    const { Parent } = getBrainWorldComponents(brainWorld);
    return hasComponent(brainWorld, nodeEid, Parent) ? Parent.id[nodeEid] : 0;
}
export function getNodeChildren(nodeEid: number, { brainWorld } = Worlds): number[] {
    const { Children } = getBrainWorldComponents(brainWorld);
    if (!hasComponent(brainWorld, nodeEid, Children)) return [];
    const count = Children.entitiesCount[nodeEid];
    const out: number[] = [];
    for (let i = 0; i < count; i++) out.push(Children.entitiesIds.get(nodeEid, i));
    return out;
}

// Resolves the turret ATOM of a hull node: the hull node's Brain child whose physics
// presentation carries VehicleTurret. Replaces the old Tank.turretEId[hullBrain] read
// (hull node IS the hull-brain). Returns 0 if there is no turret child (yet).
export function getTurretPhysOfHull(hullNode: number, worlds = Worlds): number {
    const { physicsWorld } = worlds;
    const { VehicleTurret } = getPhysicsWorldComponents(physicsWorld);
    for (const childNode of getNodeChildren(hullNode, worlds)) {
        const atom = getNodePhysics(childNode, worlds);
        if (atom !== 0 && hasComponent(physicsWorld, atom, VehicleTurret)) return atom;
    }
    return 0;
}

// Resolves the brain node whose presentation is the given physics atom (reverse of
// getNodePhysics, by a query over drawn/headless nodes — no reverse ref on the atom).
// Cold lookup: spawn-time slot attachment, teardown, and the once-per-tick resolution
// of an action's hull-atom owner / a sound's owner atom -> its node.
export function getNodeByPhysics(physEid: number, worlds = Worlds): number {
    const { brainWorld } = worlds;
    const { NodeRenderRef, NodePhysicsRef } = getBrainWorldComponents(brainWorld);
    for (const nodeEid of query(brainWorld, [NodePhysicsRef.ref])) {
        if (NodePhysicsRef.id[nodeEid] === physEid) return nodeEid;
    }
    for (const nodeEid of query(brainWorld, [NodeRenderRef.ref])) {
        if (getPhysicsOf(NodeRenderRef.id[nodeEid], worlds) === physEid) return nodeEid;
    }
    return 0;
}

// Resolves the carrier NODE that owns the slot occupied by a given part atom: walks
// the slot-bearing nodes top-down (node -> NodeSlotsRef -> slot -> OccupantRef) and
// returns the node whose slot holds the part, plus that slot. Replaces the old
// part -> HomeSlotRef -> slot -> CarrierRef -> carrier-atom upward chain. Returns
// [0, 0] if the part occupies no slot. Cold: only when a part is hit/destroyed.
export function getCarrierNodeOfPart(partPhysEid: number, worlds = Worlds): [carrierNode: number, slotEid: number] {
    const { brainWorld } = worlds;
    const { NodeSlotsRef } = getBrainWorldComponents(brainWorld);
    for (const nodeEid of query(brainWorld, [NodeSlotsRef.ref])) {
        for (const slotEid of getNodeSlots(nodeEid, worlds)) {
            if (getOccupantOf(slotEid, worlds) === partPhysEid) return [nodeEid, slotEid];
        }
    }
    return [0, 0];
}

// ---- sound (sound -> owner atom) ----
export function setSoundOwner(soundEid: number, ownerAtomEid: number, { soundWorld } = Worlds) { getSoundWorldComponents(soundWorld).SoundOwnerRef.set(soundWorld, soundEid, ownerAtomEid); }
export function clearSoundOwner(soundEid: number, { soundWorld } = Worlds) { getSoundWorldComponents(soundWorld).SoundOwnerRef.clear(soundWorld, soundEid); }
export function getSoundOwnerOf(soundEid: number, { soundWorld } = Worlds): number { return getSoundWorldComponents(soundWorld).SoundOwnerRef.id[soundEid]; }
// reverse 1:N (owner -> its sounds): query (used only at teardown). Returns an array.
export function getSoundsOfOwner(ownerAtomEid: number, { soundWorld } = Worlds): number[] {
    const { SoundOwnerRef } = getSoundWorldComponents(soundWorld);
    const out: number[] = [];
    for (const soundEid of query(soundWorld, [SoundOwnerRef])) if (SoundOwnerRef.id[soundEid] === ownerAtomEid) out.push(soundEid);
    return out;
}
