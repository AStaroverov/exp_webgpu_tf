import { EntityId, hasComponent, removeComponent } from 'bitecs';
import { min, smoothstep } from '../../../../../../../lib/math.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { removePhysicalJoint } from '../../../Physical/removePhysicalJoint.ts';
import { setPhysicalCollisionGroup } from '../../../Physical/setPhysicalCollisionGroup.ts';
import { Children } from '../../Components/Children.ts';
import { Debris } from '../../Components/Debris.ts';
import { Parent } from '../../Components/Parent.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Tank } from '../../Components/Tank.ts';
import { Vehicle } from '../../Components/Vehicle.ts';
import { VehiclePart, VehiclePartCaterpillar } from '../../Components/VehiclePart.ts';
import { Joint } from '../../Components/Joint.ts';
import { TeamRef } from '../../Components/TeamRef.ts';
import { recursiveTypicalRemoveEntity, scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { spawnExplosion } from '../Explosion.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { getFilledSlotCount, getSlotCount } from '../Vehicle/VehicleParts.ts';
import { applyExplosionImpulse } from '../../../Physical/applyExplosionImpulse.ts';
import { getSlotFillerEid, isSlot, isSlotEmpty } from '../../Utils/SlotUtils.ts';
import { EngineLabels, EngineType } from '../../../Config/vehicles.ts';


export function destroyTank(vehicleEid: EntityId) {
    // Get explosion center position
    const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleEid);
    const explosionX = getMatrixTranslationX(vehicleMatrix);
    const explosionY = getMatrixTranslationY(vehicleMatrix);

    // Spawn explosion at vehicle position
    spawnExplosion({
        x: explosionX,
        y: explosionY,
        size: 60,
        duration: 1500,
    });

    // Collect all parts before tearing them off
    const partsToExplode: EntityId[] = [];

    // turret parts
    const turretEid = Tank.turretEId[vehicleEid];
    for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
        const slotEid = Children.entitiesIds.get(turretEid, i);
        if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;
        const partEid = getSlotFillerEid(slotEid);
        if (partEid === 0) continue;
        partsToExplode.push(partEid);
        tearOffTankPart(partEid);
    }

    // vehicle parts
    for (let i = 0; i < Children.entitiesCount[vehicleEid]; i++) {
        const slotEid = Children.entitiesIds.get(vehicleEid, i);
        if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;
        const partEid = getSlotFillerEid(slotEid);
        if (partEid === 0) continue;
        partsToExplode.push(partEid);
        tearOffTankPart(partEid);
    }

    for (const partEid of partsToExplode) {
        applyExplosionImpulse(partEid, explosionX, explosionY);
    }

    scheduleRemoveEntity(vehicleEid);
    scheduleRemoveEntity(turretEid);
}

export function syncRemoveTank(tankEid: EntityId) {
    recursiveTypicalRemoveEntity(tankEid);
}

export function tearOffTankPart(vehiclePartEid: number, shouldBreakConnection: boolean = true, { world } = GameDI) {
    removeComponent(world, vehiclePartEid, TeamRef);
    removeComponent(world, vehiclePartEid, PlayerRef);

    const slotEid = Parent.id[vehiclePartEid];
    
    if (shouldBreakConnection && isSlot(slotEid)) {
        Children.removeChild(slotEid, vehiclePartEid);
    }

    const jointPid = hasComponent(world, vehiclePartEid, Joint) ? Joint.pid[vehiclePartEid] : 0;
    if (jointPid > 0) {
        Joint.removeComponent(world, vehiclePartEid);
        if (hasComponent(world, vehiclePartEid, VehiclePart)) {
            VehiclePart.removeComponent(world, vehiclePartEid);
        }
        resetVehiclePartJointComponent(vehiclePartEid);
        // @todo: remove bullet collision in game, keep only for training
        setPhysicalCollisionGroup(vehiclePartEid, CollisionGroup.ALL & ~CollisionGroup.VEHICALE_BASE & ~CollisionGroup.BULLET);
        removePhysicalJoint(jointPid);
        // changePhysicalDensity(vehiclePartEid, 8);
    }

    if (!hasComponent(world, vehiclePartEid, Debris)) {
        Debris.addComponent(world, vehiclePartEid);
    }
}

export function resetVehiclePartJointComponent(vehiclePartEid: number, { world } = GameDI) {
    Joint.resetComponent(vehiclePartEid);
    VehiclePartCaterpillar.removeComponent(world, vehiclePartEid);
}

export function getTankCurrentPartsCount(vehicleEid: number) {
    const turretEid = Tank.turretEId[vehicleEid];
    return getFilledSlotCount(vehicleEid) + getFilledSlotCount(turretEid);
}

export function getTankTotalSlotCount(vehicleEid: number) {
    const turretEid = Tank.turretEId[vehicleEid];
    return getSlotCount(vehicleEid) + getSlotCount(turretEid);
}

export const HEALTH_THRESHOLD = 0.85;

export function getTankHealthAbs(tankEid: number): number {
    const health = getTankHealth(tankEid);
    const totalSlots = getTankTotalSlotCount(tankEid);
    const absHealth = health * totalSlots;
    return absHealth;
}

export function getTankHealth(tankEid: number): number {
    const totalSlots = getTankTotalSlotCount(tankEid);
    const filledSlots = getTankCurrentPartsCount(tankEid);
    const absHealth = totalSlots > 0 && filledSlots > 0 ? min(1, filledSlots / totalSlots) : 0;
    const health = smoothstep(HEALTH_THRESHOLD, 1, absHealth);

    return health;
}

export function getTankTeamId(tankEid: number) {
    const teamId = TeamRef.id[tankEid];
    return teamId;
}

export function getTankEngineLabel(vehicleEid: number): string {
    const engine = Vehicle.engineType[vehicleEid] as EngineType;
    return EngineLabels[engine];
}

