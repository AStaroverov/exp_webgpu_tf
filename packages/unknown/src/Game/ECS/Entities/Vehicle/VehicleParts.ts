import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getSlotWorldComponents, SlotWorld } from '../../createSlotWorld.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { spawnRectanglePart } from '../spawnPart.ts';
import { attachOccupant, attachSlotToNode, clearOccupant, getNodeByPhysics, getNodeParent, getNodePhysics, getNodeSlots, getTurretPhysOfHull } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { defaultVehicleOptions, VehicleOptions } from './Options.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addEntity, EntityId, hasComponent, query } from 'bitecs';
import { cos, min, sin } from '../../../../../../../lib/math.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { getSlotPartConfig, SlotPartType } from '../../Components/SlotConfig.ts';
import { isSlotEmpty, isSlotFilled } from '../../Utils/SlotUtils.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { removePhysicalJoint } from '../../../Physical/removePhysicalJoint.ts';
import { setPhysicalCollisionGroup } from '../../../Physical/setPhysicalCollisionGroup.ts';

export type PartsData = [x: number, y: number, w: number, h: number];

export function createRectangleSet(
    cols: number, rows: number,
    width: number, paddingWidth: number,
    height = width, paddingHeight = paddingWidth,
): PartsData[] {
    const count = cols * rows;
    return Array.from({ length: count }, (_, i) => {
        return [
            i * paddingWidth % (paddingWidth * cols) - (paddingWidth * cols / 2 - width / 2),
            Math.floor(i / cols) * paddingHeight - (paddingHeight * rows / 2 - height / 2),
            width, height,
        ];
    });
}

// Creates SlotWorld slots for a carrier PHYSICS atom. Each slot captures a darkened
// copy of `color` (replacing the old render-Color + updateSlotsBrightness pass). The
// carrier's brain node owns the slot list downward via NodeSlotsRef.
export function createSlotEntities(
    carrierPhysEid: EntityId,
    params: PartsData[],
    color: TColor,
    partType: SlotPartType,
    { slotWorld } = Worlds,
) {
    const { Slot, Color } = getSlotWorldComponents(slotWorld);
    const darkened = new Float32Array(4);

    // The carrier's brain node owns the slot list downward (NodeSlotsRef). Resolved
    // once for the whole set.
    const carrierNode = getNodeByPhysics(carrierPhysEid);

    for (let i = 0; i < params.length; i++) {
        const slotEid = addEntity(slotWorld);
        const param = params[i];
        const x = param[0];
        const y = param[1];
        const width = param[2];
        const height = param[3];

        adjustBrightness(color, darkened, i / params.length / 2 - 0.1, i / params.length / 2 + 0.1);

        Slot.addComponent(slotWorld, slotEid, x, y, width, height, partType);
        Color.addComponent(slotWorld, slotEid, darkened[0], darkened[1], darkened[2], darkened[3]);
        if (carrierNode !== 0) attachSlotToNode(carrierNode, slotEid);
    }
}

export function fillAllSlots(carrierPhysEid: EntityId, options: VehicleOptions, { slotWorld } = Worlds): void {
    const carrierNode = getNodeByPhysics(carrierPhysEid);
    for (const slotEid of getNodeSlots(carrierNode)) {
        if (isSlotEmpty(slotWorld, slotEid)) {
            fillSlot(slotEid, carrierNode, options);
        }
    }
}

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);
const fillSlotOptions: VehicleOptions = structuredClone(defaultVehicleOptions);
export function fillSlot(
    slotEid: EntityId,
    carrierNode: EntityId,
    options: VehicleOptions,
    { slotWorld, physicsWorld, physicalWorld, brainWorld } = Worlds,
) {
    const { Slot } = getSlotWorldComponents(slotWorld);
    const {
        RigidBodyRef, VehiclePart, Joint, TeamRef, PlayerRef,
        Hitable, Damagable, VehiclePartCaterpillar,
    } = getPhysicsWorldComponents(physicsWorld);
    const { Vehicle } = getBrainWorldComponents(brainWorld);

    if (isNaN(options.x) || isNaN(options.y) || isNaN(options.rotation)) {
        throw new Error('Some options are not set');
    }
    if (isSlotFilled(slotWorld, slotEid)) {
        return;
    }
    Object.assign(fillSlotOptions, options);

    // The carrier node owns the slot. Its presentation (downward) is the carrier atom
    // (the joint parent body). Resolve the vehicle (for the config lookup): if the
    // carrier node IS the hull node it carries Vehicle; otherwise it's a turret/gun/
    // track node, so its Brain parent (the hull node) carries Vehicle.
    const carrierPhysEid = getNodePhysics(carrierNode);
    let vehicleBrain = carrierNode;
    if (!hasComponent(brainWorld, vehicleBrain, Vehicle)) {
        vehicleBrain = getNodeParent(carrierNode);
    }
    const vehicleType = Vehicle.type[vehicleBrain] as VehicleType;

    const partType = Slot.partType[slotEid] as SlotPartType;
    const config = getSlotPartConfig(partType, vehicleType);
    const anchorX = Slot.anchorX[slotEid];
    const anchorY = Slot.anchorY[slotEid];

    const worldX = anchorX * cos(fillSlotOptions.rotation) - anchorY * sin(fillSlotOptions.rotation);
    const worldY = anchorX * sin(fillSlotOptions.rotation) + anchorY * cos(fillSlotOptions.rotation);

    fillSlotOptions.x += worldX;
    fillSlotOptions.y += worldY;
    fillSlotOptions.z = config.z;
    fillSlotOptions.width = Slot.width[slotEid];
    fillSlotOptions.height = Slot.height[slotEid];

    fillSlotOptions.density = config.density;
    fillSlotOptions.belongsSolverGroup = config.belongsSolverGroup;
    fillSlotOptions.interactsSolverGroup = config.interactsSolverGroup;
    fillSlotOptions.belongsCollisionGroup = config.belongsCollisionGroup;
    fillSlotOptions.interactsCollisionGroup = config.interactsCollisionGroup;
    fillSlotOptions.color = readSlotColor(slotWorld, slotEid, new Float32Array(4));

    const rbId = RigidBodyRef.id[carrierPhysEid];

    const [partPhysEid, , pid] = spawnRectanglePart(fillSlotOptions);

    jointParentAnchor.x = anchorX;
    jointParentAnchor.y = anchorY;

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(jointParentAnchor, 0, jointChildAnchor, 0),
        physicalWorld.getRigidBody(rbId),
        physicalWorld.getRigidBody(pid),
        false,
    );

    VehiclePart.addComponent(physicsWorld, partPhysEid);
    Joint.addComponent(physicsWorld, partPhysEid, joint.handle);

    // Cheap static team/player copy for the cold contact path (saveHitters).
    TeamRef.addComponent(physicsWorld, partPhysEid, fillSlotOptions.teamId);
    PlayerRef.addComponent(physicsWorld, partPhysEid, fillSlotOptions.playerId);
    Hitable.addComponent(physicsWorld, partPhysEid, min(fillSlotOptions.width, fillSlotOptions.height));
    Damagable.addComponent(physicsWorld, partPhysEid, min(fillSlotOptions.width, fillSlotOptions.height) / 20);

    // slot -> part edge: OccupantRef on the slot (downward; the part keeps no ref up).
    attachOccupant(slotEid, partPhysEid);

    if (partType === SlotPartType.Caterpillar) {
        VehiclePartCaterpillar.addComponent(physicsWorld, partPhysEid);
    }
}

export function getEmptySlotsCount(carrierPhysEid: EntityId, { slotWorld } = Worlds): number {
    let emptyCount = 0;
    for (const slotEid of getNodeSlots(getNodeByPhysics(carrierPhysEid))) {
        if (isSlotEmpty(slotWorld, slotEid)) emptyCount++;
    }
    return emptyCount;
}

export function findFirstEmptySlot(carrierPhysEid: EntityId, { slotWorld } = Worlds): EntityId | null {
    for (const slotEid of getNodeSlots(getNodeByPhysics(carrierPhysEid))) {
        if (isSlotEmpty(slotWorld, slotEid)) return slotEid;
    }
    return null;
}

export function getSlotCount(carrierPhysEid: EntityId): number {
    let count = 0;
    for (const _slotEid of getNodeSlots(getNodeByPhysics(carrierPhysEid))) count++;
    return count;
}

export function getFilledSlotCount(carrierPhysEid: EntityId, { slotWorld } = Worlds): number {
    let filled = 0;
    for (const slotEid of getNodeSlots(getNodeByPhysics(carrierPhysEid))) {
        if (isSlotFilled(slotWorld, slotEid)) filled++;
    }
    return filled;
}

export function getVehicleTotalSlotCount(vehiclePhysEid: EntityId): number {
    const turretPhysEid = getTurretPhysOfHull(getNodeByPhysics(vehiclePhysEid));
    return getSlotCount(vehiclePhysEid) + getSlotCount(turretPhysEid);
}

export function getVehicleFilledSlotCount(vehiclePhysEid: EntityId): number {
    const turretPhysEid = getTurretPhysOfHull(getNodeByPhysics(vehiclePhysEid));
    return getFilledSlotCount(vehiclePhysEid) + getFilledSlotCount(turretPhysEid);
}

// Finds the slot currently occupied by a part atom (slot -> part is OccupantRef,
// downward; this is the reverse, resolved by a query over filled slots — no reverse
// ref on the part). Returns 0 if the part occupies no slot (already torn off). Cold:
// only used when a part is destroyed/explodes.
export function findSlotOfOccupant(partPhysEid: number, { slotWorld } = Worlds): number {
    const { OccupantRef } = getSlotWorldComponents(slotWorld);
    for (const slotEid of query(slotWorld, [OccupantRef])) {
        if (OccupantRef.id[slotEid] === partPhysEid) return slotEid;
    }
    return 0;
}

// Detaches a part atom from its slot: clears team/player, removes the joint, resets
// collision group. The slot's OccupantRef is cleared (slot becomes empty); the part
// itself keeps no slot reference (a torn part still lies on the ground).
export function detachPart(
    partPhysEid: number,
    shouldBreakConnection: boolean = true,
    { physicsWorld, physicalWorld } = Worlds,
) {
    const { TeamRef, PlayerRef, Joint, VehiclePart, VehiclePartCaterpillar } = getPhysicsWorldComponents(physicsWorld);

    // A detached part stops being credited: drop its static team/player copy.
    if (hasComponent(physicsWorld, partPhysEid, TeamRef)) {
        TeamRef.removeComponent(physicsWorld, partPhysEid);
        PlayerRef.removeComponent(physicsWorld, partPhysEid);
    }

    if (shouldBreakConnection) {
        const slotEid = findSlotOfOccupant(partPhysEid);
        if (slotEid !== 0) {
            clearOccupant(slotEid);
        }
    }

    const jointPid = hasComponent(physicsWorld, partPhysEid, Joint) ? Joint.pid[partPhysEid] : 0;
    if (jointPid > 0) {
        Joint.removeComponent(physicsWorld, partPhysEid);
        if (hasComponent(physicsWorld, partPhysEid, VehiclePart)) {
            VehiclePart.removeComponent(physicsWorld, partPhysEid);
        }
        Joint.resetComponent(partPhysEid);
        VehiclePartCaterpillar.removeComponent(physicsWorld, partPhysEid);
        setPhysicalCollisionGroup(physicsWorld, physicalWorld, partPhysEid, CollisionGroup.ALL & ~CollisionGroup.VEHICALE_BASE & ~CollisionGroup.BULLET);
        removePhysicalJoint(physicalWorld, jointPid);
    }
}

function readSlotColor<T extends TColor>(slotWorld: SlotWorld, slotEid: EntityId, out: T): T {
    const { Color } = getSlotWorldComponents(slotWorld);
    return Color.applyColorToArray(slotEid, out);
}

// Port of the old render-Color darkening: writes a darkened copy of `src` into `out`.
function adjustBrightness(src: TColor, out: Float32Array, start: number, end: number) {
    const factor = -1 * randomRangeFloat(start, end);
    out[0] = clamp(src[0] + factor, 0, 1);
    out[1] = clamp(src[1] + factor, 0, 1);
    out[2] = clamp(src[2] + factor, 0, 1);
    out[3] = src[3];
}
