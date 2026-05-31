import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { getRenderWorldComponents, RenderGameWorld } from '../../createRenderWorld.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
import { spawnRectanglePart, SpawnCtx } from '../spawnPart.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { defaultVehicleOptions, VehicleOptions } from './Options.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addEntity, EntityId, hasComponent } from 'bitecs';
import { cos, min, sin } from '../../../../../../../lib/math.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { getSlotPartConfig, SlotPartType } from '../../Components/SlotConfig.ts';
import { isSlot, isSlotEmpty, isSlotFilled } from '../../Utils/SlotUtils.ts';

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

// `renderWorld` carrier render eid (slots are RenderWorld entities).
export function updateSlotsBrightness(renderWorld: RenderGameWorld, parentRenderEid: EntityId) {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];

    for (let i = 0; i < childCount; i++) {
        const slotEid = Children.entitiesIds.get(parentRenderEid, i);
        if (!isSlot(renderWorld, slotEid)) continue;
        adjustBrightness(renderWorld, slotEid, i / childCount / 2 - 0.1, i / childCount / 2 + 0.1);
    }
}

export function createSlotEntities(
    renderWorld: RenderGameWorld,
    parentRenderEid: EntityId,
    params: PartsData[],
    color: TColor,
    partType: SlotPartType,
) {
    const { Slot, Color, Children, Parent } = getRenderWorldComponents(renderWorld);

    for (let i = 0; i < params.length; i++) {
        const slotEid = addEntity(renderWorld);
        const param = params[i];
        const x = param[0];
        const y = param[1];
        const width = param[2];
        const height = param[3];

        Slot.addComponent(renderWorld, slotEid, x, y, width, height, partType);
        Color.addComponent(renderWorld, slotEid, color[0], color[1], color[2], color[3]);
        Children.addComponent(renderWorld, slotEid);
        Parent.addComponent(renderWorld, slotEid, parentRenderEid);

        Children.addChildren(parentRenderEid, slotEid);
    }
}

export function fillAllSlots(renderWorld: RenderGameWorld, physicalWorld: PhysicalWorld, parentRenderEid: EntityId, options: VehicleOptions): void {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];

    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentRenderEid, i);
        if (isSlot(renderWorld, childEid) && isSlotEmpty(renderWorld, childEid)) {
            fillSlot(renderWorld, physicalWorld, childEid, options);
        }
    }
}

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);
const fillSlotOptions: VehicleOptions = structuredClone(defaultVehicleOptions);
export function fillSlot(
    renderWorld: RenderGameWorld,
    physicalWorld: PhysicalWorld,
    slotEid: EntityId,
    options: VehicleOptions,
) {
    const physicsWorld = Worlds.physicsWorld;
    const { Slot, Parent, Children, Color } = getRenderWorldComponents(renderWorld);
    const {
        Vehicle, RigidBodyRef, VehiclePart, Joint, PlayerRef, TeamRef,
        Hitable, Damagable, VehiclePartCaterpillar,
    } = getPhysicsWorldComponents(physicsWorld);

    if (isNaN(options.x) || isNaN(options.y) || isNaN(options.rotation)) {
        throw new Error('Some options are not set');
    }
    if (isSlotFilled(renderWorld, slotEid)) {
        return;
    }
    Object.assign(fillSlotOptions, options);

    // Slot's carrier render eid; the vehicle (brain) render eid is its parent if the
    // carrier is a turret/track. Translate carrier render -> physics for the joint body.
    const carrierRenderEid = Parent.id[slotEid];
    const carrierPhysEid = BridgeDI.getPhysicsOf(carrierRenderEid);
    const vehiclePhysEid = hasComponent(physicsWorld, carrierPhysEid, Vehicle)
        ? carrierPhysEid
        : BridgeDI.getPhysicsOf(Parent.id[carrierRenderEid]);
    const vehicleType = Vehicle.type[vehiclePhysEid] as VehicleType;

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
    fillSlotOptions.color = Color.applyColorToArray(slotEid, new Float32Array(4));

    const rbId = RigidBodyRef.id[carrierPhysEid];

    const ctx: SpawnCtx = { physicsWorld, renderWorld, physicalWorld };
    const [partPhysEid, partRenderEid, pid] = spawnRectanglePart(ctx, fillSlotOptions);

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

    PlayerRef.addComponent(physicsWorld, partPhysEid, fillSlotOptions.playerId);
    TeamRef.addComponent(physicsWorld, partPhysEid, fillSlotOptions.teamId);
    Hitable.addComponent(physicsWorld, partPhysEid, min(fillSlotOptions.width, fillSlotOptions.height));
    Damagable.addComponent(physicsWorld, partPhysEid, min(fillSlotOptions.width, fillSlotOptions.height) / 20);

    // slot -> part edge lives in RenderWorld (slot is a RenderWorld entity).
    Parent.addComponent(renderWorld, partRenderEid, slotEid);
    Children.addChildren(slotEid, partRenderEid);

    if (partType === SlotPartType.Caterpillar) {
        VehiclePartCaterpillar.addComponent(physicsWorld, partPhysEid);
    }
}

export function getEmptySlotsCount(renderWorld: RenderGameWorld, parentRenderEid: EntityId): number {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];
    let emptyCount = 0;

    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentRenderEid, i);
        if (!isSlot(renderWorld, childEid)) continue;

        if (isSlotEmpty(renderWorld, childEid)) {
            emptyCount++;
        }
    }

    return emptyCount;
}

export function findFirstEmptySlot(renderWorld: RenderGameWorld, parentRenderEid: EntityId): EntityId | null {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentRenderEid, i);
        if (isSlot(renderWorld, childEid) && isSlotEmpty(renderWorld, childEid)) {
            return childEid;
        }
    }
    return null;
}

export function getSlotCount(renderWorld: RenderGameWorld, parentRenderEid: EntityId): number {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];
    let count = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentRenderEid, i);
        if (isSlot(renderWorld, childEid)) count++;
    }
    return count;
}

export function getFilledSlotCount(renderWorld: RenderGameWorld, parentRenderEid: EntityId): number {
    const { Children } = getRenderWorldComponents(renderWorld);
    const childCount = Children.entitiesCount[parentRenderEid];
    let filled = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentRenderEid, i);
        if (isSlot(renderWorld, childEid) && isSlotFilled(renderWorld, childEid)) filled++;
    }
    return filled;
}

// vehicleEid here is the PHYS brain eid; turret stored as phys eid; translate to render for slot walk.
export function getVehicleTotalSlotCount(vehiclePhysEid: EntityId, { physicsWorld, renderWorld } = Worlds): number {
    const { Tank } = getPhysicsWorldComponents(physicsWorld);
    const turretPhysEid = Tank.turretEId[vehiclePhysEid];
    return getSlotCount(renderWorld, BridgeDI.getRenderOf(vehiclePhysEid))
        + getSlotCount(renderWorld, BridgeDI.getRenderOf(turretPhysEid));
}

export function getVehicleFilledSlotCount(vehiclePhysEid: EntityId, { physicsWorld, renderWorld } = Worlds): number {
    const { Tank } = getPhysicsWorldComponents(physicsWorld);
    const turretPhysEid = Tank.turretEId[vehiclePhysEid];
    return getFilledSlotCount(renderWorld, BridgeDI.getRenderOf(vehiclePhysEid))
        + getFilledSlotCount(renderWorld, BridgeDI.getRenderOf(turretPhysEid));
}

function adjustBrightness(renderWorld: RenderGameWorld, eid: EntityId, start: number, end: number) {
    const { Color } = getRenderWorldComponents(renderWorld);
    const factor = -1 * randomRangeFloat(start, end);
    Color.set$(
        eid,
        clamp(Color.getR(eid) + factor, 0, 1),
        clamp(Color.getG(eid) + factor, 0, 1),
        clamp(Color.getB(eid) + factor, 0, 1),
        Color.getA(eid),
    );
}
