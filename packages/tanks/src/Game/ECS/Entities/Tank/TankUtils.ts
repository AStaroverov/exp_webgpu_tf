import { EntityId, hasComponent, removeComponent } from 'bitecs';
import { min, smoothstep } from '../../../../../../../lib/math.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { removePhysicalJoint } from '../../../Physical/removePhysicalJoint.ts';
import { setPhysicalCollisionGroup } from '../../../Physical/setPhysicalCollisionGroup.ts';
import { Children } from '../../Components/Children.ts';
import { Debris } from '../../Components/Debris.ts';
import { Parent } from '../../Components/Parent.ts';
import { Slot } from '../../Components/Slot.ts';
import { PlayerRef } from '../../Components/PlayerRef.ts';
import { Score } from '../../Components/Score.ts';
import { Tank } from '../../Components/Tank.ts';
import { TankPart, TankPartCaterpillar } from '../../Components/TankPart.ts';
import { TeamRef } from '../../Components/TeamRef.ts';
import { mapTankEngineLabel, TankEngineType } from '../../Systems/Tank/TankControllerSystems.ts';
import { recursiveTypicalRemoveEntity, scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { spawnExplosion } from '../Explosion.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { macroTasks } from '../../../../../../../lib/TasksScheduler/macroTasks.ts';
import { getFilledSlotCount, getSlotCount } from './Common/TankParts.ts';
import { DestroyByTimeout } from '../../Components/Destroy.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { applyExplosionImpulse } from '../../../Physical/applyExplosionImpulse.ts';


export function destroyTank(tankEid: EntityId) {
    // Get explosion center position
    const tankMatrix = GlobalTransform.matrix.getBatch(tankEid);
    const explosionX = getMatrixTranslationX(tankMatrix);
    const explosionY = getMatrixTranslationY(tankMatrix);

    // Spawn explosion at tank position
    spawnExplosion({
        x: explosionX,
        y: explosionY,
        size: 60,
        duration: 1500,
    });

    // Collect all parts before tearing them off
    const partsToExplode: EntityId[] = [];

    // turret parts
    const turretEid = Tank.turretEId[tankEid];
    for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
        const slotEid = Children.entitiesIds.get(turretEid, i);
        if (Slot.isEmpty(slotEid)) continue;
        const partEid = Slot.getFillerEid(slotEid);
        partsToExplode.push(partEid);
        tearOffTankPart(partEid);
    }

    // tank parts
    for (let i = 0; i < Children.entitiesCount[tankEid]; i++) {
        const slotEid = Children.entitiesIds.get(tankEid, i);
        if (Slot.isEmpty(slotEid)) continue;
        const partEid = Slot.getFillerEid(slotEid);
        partsToExplode.push(partEid);
        tearOffTankPart(partEid);
    }

    macroTasks.addTimeout(() => {
        // Apply explosion impulse to all parts after they're detached
        for (const partEid of partsToExplode) {
            applyExplosionImpulse(partEid, explosionX, explosionY);
        }
    }, 30); // delay to ensure parts are detached

    scheduleRemoveEntity(tankEid);
    scheduleRemoveEntity(turretEid);
}

export function syncRemoveTank(tankEid: EntityId) {
    recursiveTypicalRemoveEntity(tankEid);
}

export function tearOffTankPart(tankPartEid: number, shouldBreakConnection: boolean = true, { world } = GameDI) {
    removeComponent(world, tankPartEid, TeamRef);
    removeComponent(world, tankPartEid, PlayerRef);

    const slotEid = Parent.id[tankPartEid];
    
    if (shouldBreakConnection && hasComponent(world, slotEid, Slot)) {
        Children.removeChild(slotEid, tankPartEid);
    }

    const jointPid = hasComponent(world, tankPartEid, TankPart) ? TankPart.jointPid[tankPartEid] : 0;
    if (jointPid > 0) {
        removeComponent(world, tankPartEid, TankPart);
        resetTankPartJointComponent(tankPartEid);
        // @todo: remove bullet collision in game, keep only for training
        setPhysicalCollisionGroup(tankPartEid, CollisionGroup.ALL & ~CollisionGroup.TANK_BASE & ~CollisionGroup.BULLET);
        removePhysicalJoint(jointPid);
        // changePhysicalDensity(tankPartEid, 8);
    }

    if (!hasComponent(world, tankPartEid, Debris)) {
        Debris.addComponent(world, tankPartEid);
    }

    if (!hasComponent(world, tankPartEid, DestroyByTimeout)) {
        DestroyByTimeout.addComponent(world, tankPartEid, 5_000 + randomRangeFloat(0, 5_000));
    }
}

export function resetTankPartJointComponent(tankPartEid: number, { world } = GameDI) {
    TankPart.resetComponent(tankPartEid);
    TankPartCaterpillar.removeComponent(world, tankPartEid);
}

export function getTankCurrentPartsCount(tankEid: number) {
    const turretEid = Tank.turretEId[tankEid];
    return getFilledSlotCount(tankEid) + getFilledSlotCount(turretEid);
}

export function getTankTotalSlotCount(tankEid: number) {
    const turretEid = Tank.turretEId[tankEid];
    return getSlotCount(tankEid) + getSlotCount(turretEid);
}

export const HEALTH_THRESHOLD = 0.85;

// return from 0 to 1
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

export function getTankScore(tankEid: number): number {
    const playerId = PlayerRef.id[tankEid];
    const score = Score.positiveScore[playerId] + Score.negativeScore[playerId] * 1.3;
    return score;
}

export function getTankTeamId(tankEid: number) {
    const teamId = TeamRef.id[tankEid];
    return teamId;
}

export function getTankEngineLabel(tankEid: number): string {
    const engine = Tank.engineType[tankEid] as TankEngineType;
    return mapTankEngineLabel[engine];
}

