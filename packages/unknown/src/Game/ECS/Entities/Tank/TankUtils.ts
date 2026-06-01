import { EntityId, query } from 'bitecs';
import { min, smoothstep } from '../../../../../../../lib/math.ts';
import { getBrainWorldComponents } from '../../createBrainWorld.ts';
import { getNodeByPhysics, getNodeRender, getNodeSlots, getOccupantOf, getTurretPhysOfHull } from '../../refs.ts';
import { Worlds } from '../../../DI/Worlds.ts';
import { recursiveTypicalRemoveEntity, scheduleRemoveEntity } from '../../Utils/typicalRemoveEntity.ts';
import { spawnExplosion } from '../Explosion.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { detachPart, getFilledSlotCount, getSlotCount } from '../Vehicle/VehicleParts.ts';
import { applyExplosionImpulse } from '../../../Physical/applyExplosionImpulse.ts';
import { isSlotEmpty } from '../../Utils/SlotUtils.ts';
import { EngineLabels, EngineType } from '../../../Config/vehicles.ts';


export function destroyTank(vehiclePhysEid: EntityId, { physicsWorld: world, slotWorld } = Worlds) {
    const vehicleRenderEid = getNodeRender(getNodeByPhysics(vehiclePhysEid));
    const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleRenderEid);
    const explosionX = getMatrixTranslationX(vehicleMatrix);
    const explosionY = getMatrixTranslationY(vehicleMatrix);

    spawnExplosion({
        x: explosionX,
        y: explosionY,
        size: 60,
        duration: 1500,
    });

    const partsToExplode: EntityId[] = [];

    const turretPhysEid = getTurretPhysOfHull(getNodeByPhysics(vehiclePhysEid));

    const collectFromCarrier = (carrierPhysEid: number) => {
        for (const slotEid of getNodeSlots(getNodeByPhysics(carrierPhysEid))) {
            if (isSlotEmpty(slotWorld, slotEid)) continue;
            const partPhysEid = getOccupantOf(slotEid);
            if (partPhysEid === 0) continue;
            partsToExplode.push(partPhysEid);
            tearOffTankPart(partPhysEid);
        }
    };

    collectFromCarrier(turretPhysEid);
    collectFromCarrier(vehiclePhysEid);

    for (const partPhysEid of partsToExplode) {
        applyExplosionImpulse(world, partPhysEid, explosionX, explosionY);
    }

    scheduleRemoveEntity(vehiclePhysEid);
    scheduleRemoveEntity(turretPhysEid);
}

export function syncRemoveTank(tankPhysEid: EntityId) {
    recursiveTypicalRemoveEntity(tankPhysEid);
}

// Thin re-export to minimize call-site churn; the detach body lives in VehicleParts.
export function tearOffTankPart(vehiclePartEid: number, shouldBreakConnection: boolean = true) {
    detachPart(vehiclePartEid, shouldBreakConnection);
}

export function getTankCurrentPartsCount(vehiclePhysEid: number) {
    const turretPhysEid = getTurretPhysOfHull(getNodeByPhysics(vehiclePhysEid));
    return getFilledSlotCount(vehiclePhysEid) + getFilledSlotCount(turretPhysEid);
}

export function getTankTotalSlotCount(vehiclePhysEid: number) {
    const turretPhysEid = getTurretPhysOfHull(getNodeByPhysics(vehiclePhysEid));
    return getSlotCount(vehiclePhysEid) + getSlotCount(turretPhysEid);
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

export function getTankTeamId(tankEid: number, { brainWorld } = Worlds) {
    const { TeamRef } = getBrainWorldComponents(brainWorld);
    return TeamRef.id[getNodeByPhysics(tankEid)];
}

export function getTankEngineLabel(vehicleEid: number, { brainWorld } = Worlds): string {
    const { Vehicle } = getBrainWorldComponents(brainWorld);
    const engine = Vehicle.engineType[getNodeByPhysics(vehicleEid)] as EngineType;
    return EngineLabels[engine];
}

// Number of distinct teams across all tank hull-brains (dead-but-must-compile;
// re-homed from TeamRef.ts to read the brain world).
export function getTeamsCount({ brainWorld } = Worlds): number {
    const { Tank, TeamRef } = getBrainWorldComponents(brainWorld);
    const tanks = query(brainWorld, [Tank]);
    return new Set(tanks.map((brainEid) => TeamRef.id[brainEid])).size;
}
