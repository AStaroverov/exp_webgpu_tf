import { addComponent, hasComponent, removeComponent, World } from 'bitecs';
import { delegate } from '../../../../renderer/src/delegate.ts';
import { NestedArray } from '../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../renderer/src/ECS/utils.ts';

// On a SlotWorld slot: slot eid -> occupying part physics atom eid (0 = empty).
export const createOccupantRefComponent = defineComponent((OccupantRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[slotEid] = partPhysEid
    return {
        id,
        ref: OccupantRef,
        set(world: World, slotEid: number, partEid: number) {
            addComponent(world, slotEid, OccupantRef);
            id[slotEid] = partEid;
        },
        clear(world: World, slotEid: number) {
            id[slotEid] = 0;
            removeComponent(world, slotEid, OccupantRef);
        },
    };
});

// On a SoundWorld sound: sound eid -> owning physics atom eid (the atom the sound follows).
export const createSoundOwnerRefComponent = defineComponent((SoundOwnerRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[soundEid] = ownerAtomPhysEid
    return {
        id,
        ref: SoundOwnerRef,
        set(world: World, soundEid: number, ownerEid: number) {
            addComponent(world, soundEid, SoundOwnerRef);
            id[soundEid] = ownerEid;
        },
        clear(world: World, soundEid: number) {
            id[soundEid] = 0;
            removeComponent(world, soundEid, SoundOwnerRef);
        },
    };
});

// On a BrainWorld node: node eid -> its render mirror eid (scheme A: a node carries
// NodeRenderRef XOR NodePhysicsRef). Present => the node is drawn.
export const createNodeRenderRefComponent = defineComponent((NodeRenderRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[nodeEid] = renderEid
    return {
        id,
        ref: NodeRenderRef,
        set(world: World, nodeEid: number, renderEid: number) {
            addComponent(world, nodeEid, NodeRenderRef);
            id[nodeEid] = renderEid;
        },
        clear(world: World, nodeEid: number) {
            id[nodeEid] = 0;
            removeComponent(world, nodeEid, NodeRenderRef);
        },
    };
});

// On a BrainWorld node: node eid -> its physics atom eid (scheme A: a node carries
// NodeRenderRef XOR NodePhysicsRef). Present => body directly, no render.
export const createNodePhysicsRefComponent = defineComponent((NodePhysicsRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[nodeEid] = physicsEid
    return {
        id,
        ref: NodePhysicsRef,
        set(world: World, nodeEid: number, physEid: number) {
            addComponent(world, nodeEid, NodePhysicsRef);
            id[nodeEid] = physEid;
        },
        clear(world: World, nodeEid: number) {
            id[nodeEid] = 0;
            removeComponent(world, nodeEid, NodePhysicsRef);
        },
    };
});

// On a BrainWorld node: node eid -> its slot list (1:N, downward to SlotWorld eids).
// Flat NestedArray + per-node count (mirrors the Children component idiom). The
// largest slot set today is ~120 (harvester hull 12x10), so MAX_SLOTS keeps margin.
const MAX_SLOTS = 256;
export const createNodeSlotsRefComponent = defineComponent((NodeSlotsRef) => {
    const slotsCount = new Float64Array(delegate.defaultSize); // slotsCount[nodeEid]
    const slotsIds = NestedArray.f64(MAX_SLOTS, delegate.defaultSize); // slotsIds[nodeEid][i] = slotEid
    return {
        slotsCount,
        slotsIds,
        ref: NodeSlotsRef,
        attach(world: World, nodeEid: number, slotEid: number) {
            if (!hasComponent(world, nodeEid, NodeSlotsRef)) {
                addComponent(world, nodeEid, NodeSlotsRef);
            }
            const len = slotsCount[nodeEid];
            if (len >= MAX_SLOTS) {
                throw new Error('Max slots per node reached');
            }
            slotsIds.set(nodeEid, len, slotEid);
            slotsCount[nodeEid] = len + 1;
        },
        get(nodeEid: number): number[] {
            const len = slotsCount[nodeEid];
            return len === 0 ? EMPTY_SLOTS : Array.from(slotsIds.getBatch(nodeEid).subarray(0, len));
        },
        clear(world: World, nodeEid: number) {
            slotsCount[nodeEid] = 0;
            slotsIds.getBatch(nodeEid).fill(0);
            removeComponent(world, nodeEid, NodeSlotsRef);
        },
    };
});

const EMPTY_SLOTS: number[] = [];

// On a RenderWorld mirror: render eid -> physics atom eid.
export const createPhysicsRefComponent = defineComponent((PhysicsRef) => {
    const id = new Float64Array(delegate.defaultSize); // id[renderEid] = physicsEid
    return {
        id,
        ref: PhysicsRef,
        set(world: World, renderEid: number, physEid: number) {
            addComponent(world, renderEid, PhysicsRef);
            id[renderEid] = physEid;
        },
        clear(world: World, renderEid: number) {
            id[renderEid] = 0;
            removeComponent(world, renderEid, PhysicsRef);
        },
    };
});
