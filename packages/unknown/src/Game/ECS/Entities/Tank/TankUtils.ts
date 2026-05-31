import { EntityId, hasComponent, removeComponent } from 'bitecs';
import { min, smoothstep } from '../../../../../../../lib/math.ts';
import { CollisionGroup } from '../../../Physical/createRigid.ts';
import { removePhysicalJoint } from '../../../Physical/removePhysicalJoint.ts';
import { setPhysicalCollisionGroup } from '../../../Physical/setPhysicalCollisionGroup.ts';
import { getPhysicsWorldComponents } from '../../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { BridgeDI } from '../../../DI/BridgeDI.ts';
import { Worlds } from '../../../DI/Worlds.ts';
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


export function destroyTank(vehiclePhysEid: EntityId, { physicsWorld: world, renderWorld } = Worlds) {
    const { Tank } = getPhysicsWorldComponents(world);
    const { Children } = getRenderWorldComponents(renderWorld);

    const vehicleRenderEid = BridgeDI.getRenderOf(vehiclePhysEid);
    const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleRenderEid);
    const explosionX = getMatrixTranslationX(vehicleMatrix);
    const explosionY = getMatrixTranslationY(vehicleMatrix);

    spawnExplosion(renderWorld, {
        x: explosionX,
        y: explosionY,
        size: 60,
        duration: 1500,
    });

    const partsToExplode: EntityId[] = [];

    const turretPhysEid = Tank.turretEId[vehiclePhysEid];
    const turretRenderEid = BridgeDI.getRenderOf(turretPhysEid);

    const collectFromCarrier = (carrierRenderEid: number) => {
        for (let i = 0; i < Children.entitiesCount[carrierRenderEid]; i++) {
            const slotEid = Children.entitiesIds.get(carrierRenderEid, i);
            if (!isSlot(renderWorld, slotEid) || isSlotEmpty(renderWorld, slotEid)) continue;
            const partRenderEid = getSlotFillerEid(renderWorld, slotEid);
            if (partRenderEid === 0) continue;
            const partPhysEid = BridgeDI.getPhysicsOf(partRenderEid);
            partsToExplode.push(partPhysEid);
            tearOffTankPart(partPhysEid);
        }
    };

    collectFromCarrier(turretRenderEid);
    collectFromCarrier(vehicleRenderEid);

    for (const partPhysEid of partsToExplode) {
        applyExplosionImpulse(world, partPhysEid, explosionX, explosionY);
    }

    scheduleRemoveEntity(vehiclePhysEid);
    scheduleRemoveEntity(turretPhysEid);
}

export function syncRemoveTank(tankPhysEid: EntityId) {
    recursiveTypicalRemoveEntity(tankPhysEid);
}

export function tearOffTankPart(vehiclePartEid: number, shouldBreakConnection: boolean = true, { physicsWorld: world, renderWorld, physicalWorld } = Worlds) {
    const { TeamRef, PlayerRef, Joint, VehiclePart } = getPhysicsWorldComponents(world);
    const { Parent, Children } = getRenderWorldComponents(renderWorld);

    removeComponent(world, vehiclePartEid, TeamRef);
    removeComponent(world, vehiclePartEid, PlayerRef);

    // The slot->part edge lives in RenderWorld; resolve via the part's mirror.
    const partRenderEid = BridgeDI.getRenderOf(vehiclePartEid);
    const slotEid = Parent.id[partRenderEid];

    if (shouldBreakConnection && isSlot(renderWorld, slotEid)) {
        Children.removeChild(slotEid, partRenderEid);
    }

    const jointPid = hasComponent(world, vehiclePartEid, Joint) ? Joint.pid[vehiclePartEid] : 0;
    if (jointPid > 0) {
        Joint.removeComponent(world, vehiclePartEid);
        if (hasComponent(world, vehiclePartEid, VehiclePart)) {
            VehiclePart.removeComponent(world, vehiclePartEid);
        }
        resetVehiclePartJointComponent(vehiclePartEid);
        setPhysicalCollisionGroup(world, physicalWorld, vehiclePartEid, CollisionGroup.ALL & ~CollisionGroup.VEHICALE_BASE & ~CollisionGroup.BULLET);
        removePhysicalJoint(physicalWorld, jointPid);
    }
}

export function resetVehiclePartJointComponent(vehiclePartEid: number, { physicsWorld: world } = Worlds) {
    const { Joint, VehiclePartCaterpillar } = getPhysicsWorldComponents(world);
    Joint.resetComponent(vehiclePartEid);
    VehiclePartCaterpillar.removeComponent(world, vehiclePartEid);
}

export function getTankCurrentPartsCount(vehiclePhysEid: number, { physicsWorld, renderWorld } = Worlds) {
    const { Tank } = getPhysicsWorldComponents(physicsWorld);
    const turretPhysEid = Tank.turretEId[vehiclePhysEid];
    return getFilledSlotCount(renderWorld, BridgeDI.getRenderOf(vehiclePhysEid))
        + getFilledSlotCount(renderWorld, BridgeDI.getRenderOf(turretPhysEid));
}

export function getTankTotalSlotCount(vehiclePhysEid: number, { physicsWorld, renderWorld } = Worlds) {
    const { Tank } = getPhysicsWorldComponents(physicsWorld);
    const turretPhysEid = Tank.turretEId[vehiclePhysEid];
    return getSlotCount(renderWorld, BridgeDI.getRenderOf(vehiclePhysEid))
        + getSlotCount(renderWorld, BridgeDI.getRenderOf(turretPhysEid));
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

export function getTankTeamId(tankEid: number, { physicsWorld: world } = Worlds) {
    const { TeamRef } = getPhysicsWorldComponents(world);
    return TeamRef.id[tankEid];
}

export function getTankEngineLabel(vehicleEid: number, { physicsWorld: world } = Worlds): string {
    const { Vehicle } = getPhysicsWorldComponents(world);
    const engine = Vehicle.engineType[vehicleEid] as EngineType;
    return EngineLabels[engine];
}
