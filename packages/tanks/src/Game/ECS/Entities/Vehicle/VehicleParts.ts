import { Color, TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../DI/GameDI.ts';
import { RigidBodyRef } from '../../Components/Physical.ts';
import { createRectangleRR } from '../../Components/RigidRender.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Hitable } from '../../Components/Hitable.ts';
import { Parent } from '../../Components/Parent.ts';
import { Children } from '../../Components/Children.ts';
import { VehiclePart, VehiclePartCaterpillar } from '../../Components/VehiclePart.ts';
import { defaultVehicleOptions, VehicleOptions } from './Options.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addComponent, addEntity, EntityId, hasComponent } from 'bitecs';
import { TeamRef } from '../../Components/TeamRef.ts';
import { Damagable } from '../../Components/Damagable.ts';
import { cos, min, sin } from '../../../../../../../lib/math.ts';
import { Slot } from '../../Components/Slot.ts';
import { Tank } from '../../Components/Tank.ts';
import { Vehicle, VehicleType } from '../../Components/Vehicle.ts';
import { VehicleTurret } from '../../Components/VehicleTurret.ts';
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

export function updateSlotsBrightness(parentEId: EntityId) {
    const childCount = Children.entitiesCount[parentEId];

    for (let i = 0; i < childCount; i++) {
        const slotEid = Children.entitiesIds.get(parentEId, i);
        if (!isSlot(slotEid)) continue;
        adjustBrightness(slotEid, i / childCount / 2 - 0.1, i / childCount / 2 + 0.1);
    }
}

export function createSlotEntities(
    parentEId: EntityId,
    params: PartsData[],
    color: TColor,
    partType: SlotPartType,
    { world } = GameDI,
) {
    for (let i = 0; i < params.length; i++) {
        const slotEid = addEntity(world);
        const param = params[i];
        const x = param[0];
        const y = param[1];
        const width = param[2];
        const height = param[3];

        Slot.addComponent(world, slotEid, x, y, width, height, partType);
        Color.addComponent(world, slotEid, color[0], color[1], color[2], color[3]);
        Children.addComponent(world, slotEid);
        Parent.addComponent(world, slotEid, parentEId);

        // Add slot as child of vehicle/turret
        Children.addChildren(parentEId, slotEid);
    }
}

/**
 * Fill all empty slots with physical parts
 * Iterates Children of parent, filters by Slot component
 */
export function fillAllSlots(parentEId: EntityId, options: VehicleOptions): void {
    const childCount = Children.entitiesCount[parentEId];
    
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (isSlot(childEid) && isSlotEmpty(childEid)) {
            fillSlot(childEid, options);
        }
    }
}

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);
const fillSlotOptions: VehicleOptions = structuredClone(defaultVehicleOptions);
export function fillSlot(
    slotEid: EntityId,
    options: VehicleOptions,
    { world, physicalWorld } = GameDI,
) {
    if (isNaN(options.x) || isNaN(options.y) || isNaN(options.rotation)) {
        throw new Error('Some options are not set');
    }
    if (isSlotFilled(slotEid)) {
        return;
    }
    Object.assign(fillSlotOptions, options);

    // Slot's parent is Vehicle or Turret (the physical body to attach to)
    const vehicleOrTurretEid = Parent.id[slotEid];
    // Get vehicle entity and type
    // Parent of slot can be Vehicle (for hull/caterpillar) or VehicleTurret (for turret parts)
    const vehicleEid = hasComponent(world, vehicleOrTurretEid, Vehicle) 
        ? vehicleOrTurretEid 
        : VehicleTurret.vehicleEId[vehicleOrTurretEid];
    const vehicleType = Vehicle.type[vehicleEid] as VehicleType;
    
    // Get config from slot's part type and vehicle type
    const partType = Slot.partType[slotEid] as SlotPartType;
    const config = getSlotPartConfig(partType, vehicleType);
    const anchorX = Slot.anchorX[slotEid];
    const anchorY = Slot.anchorY[slotEid];

    // Transform anchor from local to world space
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
    fillSlotOptions.shadow[1] = config.shadowY;
    fillSlotOptions.color = Color.applyColorToArray(slotEid, new Float32Array(4));
    // Create the physical part at the correct world position
    const rbId = RigidBodyRef.id[vehicleOrTurretEid];
    const [eid, pid] = createRectangleRR(fillSlotOptions);

    jointParentAnchor.x = anchorX;
    jointParentAnchor.y = anchorY;

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(jointParentAnchor, 0, jointChildAnchor, 0),
        physicalWorld.getRigidBody(rbId),
        physicalWorld.getRigidBody(pid),
        false,
    );
    
    VehiclePart.addComponent(world, eid, joint.handle);

    PlayerRef.addComponent(world, eid, fillSlotOptions.playerId);
    TeamRef.addComponent(world, eid, fillSlotOptions.teamId);
    Hitable.addComponent(world, eid, min(fillSlotOptions.width, fillSlotOptions.height));
    Damagable.addComponent(world, eid, min(fillSlotOptions.width, fillSlotOptions.height) / 20);
    
    // VehiclePart is child of Slot (not vehicle/turret directly)
    Parent.addComponent(world, eid, slotEid);
    Children.addChildren(slotEid, eid);

    if (partType === SlotPartType.Caterpillar) {
        addComponent(world, eid, VehiclePartCaterpillar);
    }
}

export function getEmptySlotsCount(parentEId: EntityId): number {
    const childCount = Children.entitiesCount[parentEId];
    let emptyCount = 0;
    
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (!isSlot(childEid)) continue;
        
        if (isSlotEmpty(childEid)) {
            emptyCount++;
        }
    }

    return emptyCount;
}

/**
 * Find first empty slot entity among children of parent
 */
export function findFirstEmptySlot(parentEId: EntityId): EntityId | null {
    const childCount = Children.entitiesCount[parentEId];
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (isSlot(childEid) && isSlotEmpty(childEid)) {
            return childEid;
        }
    }
    return null;
}

/**
 * Get slot count for a parent (vehicle or turret)
 */
export function getSlotCount(parentEId: EntityId): number {
    const childCount = Children.entitiesCount[parentEId];
    let count = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (isSlot(childEid)) count++;
    }
    return count;
}

/**
 * Get filled slot count for a parent (vehicle or turret)
 */
export function getFilledSlotCount(parentEId: EntityId): number {
    const childCount = Children.entitiesCount[parentEId];
    let filled = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (isSlot(childEid) && isSlotFilled(childEid)) filled++;
    }
    return filled;
}

/**
 * Get total slot count for a vehicle (hull + turret)
 */
export function getVehicleTotalSlotCount(vehicleEid: EntityId): number {
    const turretEid = Tank.turretEId[vehicleEid];
    return getSlotCount(vehicleEid) + getSlotCount(turretEid);
}

/**
 * Get filled slot count for a vehicle (hull + turret)
 */
export function getVehicleFilledSlotCount(vehicleEid: EntityId): number {
    const turretEid = Tank.turretEId[vehicleEid];
    return getFilledSlotCount(vehicleEid) + getFilledSlotCount(turretEid);
}

function adjustBrightness(eid: EntityId, start: number, end: number) {
    const factor = -1 * randomRangeFloat(start, end);
    Color.r[eid] = clamp(Color.r[eid] + factor, 0, 1);
    Color.g[eid] = clamp(Color.g[eid] + factor, 0, 1);
    Color.b[eid] = clamp(Color.b[eid] + factor, 0, 1);
}

