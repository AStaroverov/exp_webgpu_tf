import { Color, TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
import { JointData, Vector2 } from '@dimforge/rapier2d-simd';
import { GameDI } from '../../../../DI/GameDI.ts';
import { RigidBodyRef } from '../../../Components/Physical.ts';
import { createRectangleRR } from '../../../Components/RigidRender.ts';
import { PlayerRef } from '../../../Components/PlayerRef.ts';
import { Hitable } from '../../../Components/Hitable.ts';
import { Parent } from '../../../Components/Parent.ts';
import { Children } from '../../../Components/Children.ts';
import { TankPart, TankPartCaterpillar } from '../../../Components/TankPart.ts';
import { Options } from './Options.ts';
import { randomRangeFloat } from '../../../../../../../../lib/random.ts';
import { clamp } from 'lodash-es';
import { addComponent, addEntity, EntityId, hasComponent } from 'bitecs';
import { TeamRef } from '../../../Components/TeamRef.ts';
import { Damagable } from '../../../Components/Damagable.ts';
import { min } from '../../../../../../../../lib/math.ts';
import { Slot } from '../../../Components/Slot.ts';
import { Tank, TankType } from '../../../Components/Tank.ts';
import { TankTurret } from '../../../Components/TankTurret.ts';
import { getSlotPartConfig, SlotPartType } from '../../../Components/SlotConfig.ts';

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

export function updateSlotsBrightness(parentEId: EntityId, { world } = GameDI) {
    const childCount = Children.entitiesCount[parentEId];

    for (let i = 0; i < childCount; i++) {
        const slotEid = Children.entitiesIds.get(parentEId, i);
        if (!hasComponent(world, slotEid, Slot)) continue;
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

        // Add slot as child of tank/turret
        Children.addChildren(parentEId, slotEid);
    }
}

/**
 * Fill all empty slots with physical parts
 * Iterates Children of parent, filters by Slot component
 */
export function fillAllSlots(parentEId: EntityId, options: Options, { world } = GameDI): void {
    const childCount = Children.entitiesCount[parentEId];
    
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (!hasComponent(world, childEid, Slot)) continue;
        
        if (Slot.isEmpty(childEid)) {
            fillSlot(childEid, options);
        }
    }
}

const jointParentAnchor = new Vector2(0, 0);
const jointChildAnchor = new Vector2(0, 0);

export function fillSlot(
    slotEid: EntityId,
    options: Options,
    { world, physicalWorld } = GameDI,
): EntityId | null {
    if (Slot.isFilled(slotEid)) {
        return null; // Slot already filled
    }

    // Slot's parent is Tank or Turret (the physical body to attach to)
    const tankOrTurretEid = Parent.id[slotEid];

    const rbId = RigidBodyRef.id[tankOrTurretEid];
    const parentRb = physicalWorld.getRigidBody(rbId);
    if (!parentRb) return null;

    // const parentTranslation = parentRb.translation();
    // const parentRotation = parentRb.rotation();
    
    // // Transform local anchor to world coordinates (rotate by parent rotation)
    const anchorX = Slot.anchorX[slotEid];
    const anchorY = Slot.anchorY[slotEid];
    // const cos = Math.cos(parentRotation);
    // const sin = Math.sin(parentRotation);
    // const worldOffsetX = anchorX * cos - anchorY * sin;
    // const worldOffsetY = anchorX * sin + anchorY * cos;
    
    // Get tank entity and type
    // Parent of slot can be Tank (for hull/caterpillar) or TankTurret (for turret parts)
    const tankEid = hasComponent(world, tankOrTurretEid, Tank) 
        ? tankOrTurretEid 
        : TankTurret.tankEId[tankOrTurretEid];
    const tankType = Tank.type[tankEid] as TankType;
    
    // Get config from slot's part type and tank type
    const partType = Slot.partType[slotEid] as SlotPartType;
    const config = getSlotPartConfig(partType, tankType);

    // options.x = parentTranslation.x// + worldOffsetX;
    // options.y = parentTranslation.y// + worldOffsetY;
    options.z = config.z;
    // options.rotation = parentRotation;
    options.width = Slot.width[slotEid];
    options.height = Slot.height[slotEid];

    options.density = config.density;
    options.belongsSolverGroup = config.belongsSolverGroup;
    options.interactsSolverGroup = config.interactsSolverGroup;
    options.belongsCollisionGroup = config.belongsCollisionGroup;
    options.interactsCollisionGroup = config.interactsCollisionGroup;
    options.shadow[1] = config.shadowY;
    options.color = Color.applyColorToArray(slotEid, new Float32Array(4));
    // Create the physical part
    const [eid, pid] = createRectangleRR(options);

    jointParentAnchor.x = anchorX;
    jointParentAnchor.y = anchorY;

    const joint = physicalWorld.createImpulseJoint(
        JointData.fixed(jointParentAnchor, 0, jointChildAnchor, 0),
        parentRb,
        physicalWorld.getRigidBody(pid),
        false,
    );
    
    // Add components
    TankPart.addComponent(world, eid, joint.handle);

    PlayerRef.addComponent(world, eid, options.playerId);
    TeamRef.addComponent(world, eid, options.teamId);
    Hitable.addComponent(world, eid, min(options.width, options.height));
    Damagable.addComponent(world, eid, min(options.width, options.height) / 20);
    
    // TankPart is child of Slot (not tank/turret directly)
    Parent.addComponent(world, eid, slotEid);
    Children.addChildren(slotEid, eid);

    if (partType === SlotPartType.Caterpillar) {
        addComponent(world, eid, TankPartCaterpillar);
    }

    return eid;
}

export function getEmptySlotsCount(parentEId: EntityId, { world } = GameDI): number {
    const childCount = Children.entitiesCount[parentEId];
    let emptyCount = 0;
    
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (!hasComponent(world, childEid, Slot)) continue;
        
        if (Slot.isEmpty(childEid)) {
            emptyCount++;
        }
    }

    return emptyCount;
}

/**
 * Find first empty slot entity among children of parent
 */
export function findFirstEmptySlot(parentEId: EntityId, { world } = GameDI): EntityId | null {
    const childCount = Children.entitiesCount[parentEId];
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (hasComponent(world, childEid, Slot) && Slot.isEmpty(childEid)) {
            return childEid;
        }
    }
    return null;
}

/**
 * Get slot count for a parent (tank or turret)
 */
export function getSlotCount(parentEId: EntityId, { world } = GameDI): number {
    const childCount = Children.entitiesCount[parentEId];
    let count = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (hasComponent(world, childEid, Slot)) count++;
    }
    return count;
}

/**
 * Get filled slot count for a parent (tank or turret)
 */
export function getFilledSlotCount(parentEId: EntityId, { world } = GameDI): number {
    const childCount = Children.entitiesCount[parentEId];
    let filled = 0;
    for (let i = 0; i < childCount; i++) {
        const childEid = Children.entitiesIds.get(parentEId, i);
        if (hasComponent(world, childEid, Slot) && Slot.isFilled(childEid)) filled++;
    }
    return filled;
}

/**
 * Get total slot count for a tank (hull + turret)
 */
export function getTankTotalSlotCount(tankEid: EntityId): number {
    const turretEid = Tank.turretEId[tankEid];
    return getSlotCount(tankEid) + getSlotCount(turretEid);
}

/**
 * Get filled slot count for a tank (hull + turret)
 */
export function getTankFilledSlotCount(tankEid: EntityId): number {
    const turretEid = Tank.turretEId[tankEid];
    return getFilledSlotCount(tankEid) + getFilledSlotCount(turretEid);
}

function adjustBrightness(eid: EntityId, start: number, end: number) {
    const factor = -1 * randomRangeFloat(start, end);
    Color.r[eid] = clamp(Color.r[eid] + factor, 0, 1);
    Color.g[eid] = clamp(Color.g[eid] + factor, 0, 1);
    Color.b[eid] = clamp(Color.b[eid] + factor, 0, 1);
}