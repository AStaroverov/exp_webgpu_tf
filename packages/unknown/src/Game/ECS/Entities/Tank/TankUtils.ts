import { EntityId, hasComponent, removeComponent } from "bitecs";
import { min, smoothstep } from "../../../../../../../lib/math.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { CollisionGroup } from "../../../Physical/createRigid.ts";
import { removePhysicalJoint } from "../../../Physical/removePhysicalJoint.ts";
import { setPhysicalCollisionGroup } from "../../../Physical/setPhysicalCollisionGroup.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import {
  recursiveTypicalRemoveEntity,
  scheduleRemoveEntity,
} from "../../Utils/typicalRemoveEntity.ts";
import { spawnExplosion } from "../Explosion.ts";
import {
  getMatrixTranslationX,
  getMatrixTranslationY,
  GlobalTransform,
} from "../../../../../../renderer/src/ECS/Components/Transform.ts";
import { getFilledSlotCount, getSlotCount } from "../Vehicle/VehicleParts.ts";
import { applyExplosionImpulse } from "../../../Physical/applyExplosionImpulse.ts";
import { getSlotFillerEid, isSlot, isSlotEmpty } from "../../Utils/SlotUtils.ts";
import { EngineLabels, EngineType } from "../../../Config/vehicles.ts";

export function destroyTank(vehicleEid: EntityId, { world } = GameDI) {
  const { Tank, Children } = getGameComponents(world);
  const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleEid);
  const explosionX = getMatrixTranslationX(vehicleMatrix);
  const explosionY = getMatrixTranslationY(vehicleMatrix);

  spawnExplosion({
    x: explosionX,
    y: explosionY,
    size: 60,
    duration: 1500,
  });

  const partsToExplode: EntityId[] = [];

  const turretEid = Tank.turretEId[vehicleEid];
  for (let i = 0; i < Children.entitiesCount[turretEid]; i++) {
    const slotEid = Children.entitiesIds.get(turretEid, i);
    if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;
    const partEid = getSlotFillerEid(slotEid);
    if (partEid === 0) continue;
    partsToExplode.push(partEid);
    tearOffTankPart(partEid);
  }

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

export function tearOffTankPart(
  vehiclePartEid: number,
  shouldBreakConnection: boolean = true,
  { world } = GameDI,
) {
  const { TeamRef, PlayerRef, Parent, Children, Joint, VehiclePart } = getGameComponents(world);
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
    setPhysicalCollisionGroup(
      vehiclePartEid,
      CollisionGroup.ALL &
        ~CollisionGroup.VEHICALE_BASE &
        ~CollisionGroup.BULLET &
        ~CollisionGroup.TANK_TURRET_HEAD_PARTS &
        ~CollisionGroup.TANK_TURRET_GUN_PARTS,
    );
    removePhysicalJoint(jointPid);
  }
}

export function resetVehiclePartJointComponent(vehiclePartEid: number, { world } = GameDI) {
  const { Joint, VehiclePartCaterpillar } = getGameComponents(world);
  Joint.resetComponent(vehiclePartEid);
  VehiclePartCaterpillar.removeComponent(world, vehiclePartEid);
}

export function getTankCurrentPartsCount(vehicleEid: number, { world } = GameDI) {
  const { Tank } = getGameComponents(world);
  const turretEid = Tank.turretEId[vehicleEid];
  return getFilledSlotCount(vehicleEid) + getFilledSlotCount(turretEid);
}

export function getTankTotalSlotCount(vehicleEid: number, { world } = GameDI) {
  const { Tank } = getGameComponents(world);
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

export function getTankTeamId(tankEid: number, { world } = GameDI) {
  const { TeamRef } = getGameComponents(world);
  return TeamRef.id[tankEid];
}

export function getTankEngineLabel(vehicleEid: number, { world } = GameDI): string {
  const { Vehicle } = getGameComponents(world);
  const engine = Vehicle.engineType[vehicleEid] as EngineType;
  return EngineLabels[engine];
}
