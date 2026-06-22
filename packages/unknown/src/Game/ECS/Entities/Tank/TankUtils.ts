import type { EntityId } from "bitecs";
import { hasComponent, removeComponent } from "bitecs";
import { min, smoothstep } from "../../../../../../../lib/math.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { CollisionGroup, createRigidRectangle } from "../../../Physical/createRigid.ts";
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
import { VFXType } from "../../Components/VFX.ts";

/** How long torn-off debris lingers on the ground before self-destructing (ms). */
const DEBRIS_LIFETIME_MS = 10_000;

export function destroyTank(vehicleEid: EntityId, { world } = GameDI) {
  const { Tank, Children } = getGameComponents(world);
  const vehicleMatrix = GlobalTransform.matrix.getBatch(vehicleEid);
  const explosionX = getMatrixTranslationX(vehicleMatrix);
  const explosionY = getMatrixTranslationY(vehicleMatrix);

  spawnExplosion({
    x: explosionX,
    y: explosionY,
    type: VFXType.Explosion,
    size: 60,
    duration: 1500,
  });

  const partsToExplode: EntityId[] = [];

  const turretEid = Tank.turretEId.get(vehicleEid);
  for (let i = 0; i < Children.entitiesCount.get(turretEid); i++) {
    const slotEid = Children.entitiesIds.get(turretEid, i);
    if (!isSlot(slotEid) || isSlotEmpty(slotEid)) continue;
    const partEid = getSlotFillerEid(slotEid);
    if (partEid === 0) continue;
    partsToExplode.push(partEid);
    tearOffTankPart(partEid);
  }

  for (let i = 0; i < Children.entitiesCount.get(vehicleEid); i++) {
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

// Debris collision group: a torn-off part stops colliding with its former team
// (hull, bullets, turret parts) and just tumbles around as wreckage.
const DEBRIS_GROUP =
  CollisionGroup.ALL &
  ~CollisionGroup.VEHICALE_BASE &
  ~CollisionGroup.BULLET &
  ~CollisionGroup.TANK_TURRET_HEAD_PARTS &
  ~CollisionGroup.TANK_TURRET_GUN_PARTS;

/**
 * Promotes a compound armor part (a collider on the owner body) into its own
 * free-flying rigid body, preserving size, density, world pose and inherited
 * velocity. After this the part owns a body, so it leaves the compound-part
 * transform path and is driven by physics like the legacy torn-off parts.
 */
function promoteCompoundPartToBody(vehiclePartEid: number, { world, physicalWorld } = GameDI) {
  const { CompoundPart, RigidBodyRef, RigidBodyState, Impulse, TorqueImpulse } =
    getGameComponents(world);

  // Idempotent: once promoted the component is gone — a second call would read a
  // zeroed colliderHandle and detach the wrong (or no) collider.
  if (!hasComponent(world, vehiclePartEid, CompoundPart)) return;

  const ownerEid = CompoundPart.ownerEid.get(vehiclePartEid);
  const colliderHandle = CompoundPart.colliderHandle.get(vehiclePartEid);

  const matrix = GlobalTransform.matrix.getBatch(vehiclePartEid);
  const x = getMatrixTranslationX(matrix);
  const y = getMatrixTranslationY(matrix);
  const rotation = Math.atan2(matrix[1], matrix[0]);

  // Inherit the owner body's velocity so the part keeps moving with the tank.
  const speedX = RigidBodyState.linvel.get(ownerEid, 0);
  const speedY = RigidBodyState.linvel.get(ownerEid, 1);
  const angularSpeed = RigidBodyState.angvel[ownerEid];

  const collider = physicalWorld.getCollider(colliderHandle);
  const half = collider.halfExtents();
  const width = half.x * 2;
  const height = half.y * 2;
  const density = collider.density();

  physicalWorld.removeCollider(collider, true);
  CompoundPart.removeComponent(world, vehiclePartEid);

  const pid = createRigidRectangle({
    x,
    y,
    rotation,
    width,
    height,
    density,
    speedX,
    speedY,
    angularSpeed,
    belongsCollisionGroup: DEBRIS_GROUP,
    interactsCollisionGroup: DEBRIS_GROUP,
  });
  RigidBodyRef.addComponent(world, vehiclePartEid, pid);
  RigidBodyState.addComponent(world, vehiclePartEid);
  Impulse.addComponent(world, vehiclePartEid);
  TorqueImpulse.addComponent(world, vehiclePartEid);
}

export function tearOffTankPart(
  vehiclePartEid: number,
  shouldBreakConnection: boolean = true,
  { world } = GameDI,
) {
  const {
    TeamRef,
    PlayerRef,
    Parent,
    Children,
    Joint,
    VehiclePart,
    CompoundPart,
    Salvage,
    DestroyByTimeout,
  } = getGameComponents(world);
  removeComponent(world, vehiclePartEid, TeamRef);
  removeComponent(world, vehiclePartEid, PlayerRef);

  // The piece is now ownerless debris — collectable scrap for a Repairer tank,
  // but only for a while: it self-destructs after DEBRIS_LIFETIME_MS.
  Salvage.addComponent(world, vehiclePartEid);
  DestroyByTimeout.addComponent(world, vehiclePartEid, DEBRIS_LIFETIME_MS);

  const slotEid = Parent.id.get(vehiclePartEid);

  if (shouldBreakConnection && isSlot(slotEid)) {
    Children.removeChild(slotEid, vehiclePartEid);
  }

  // Compound parts have no joint — detach the collider from the owner body and
  // turn the part into standalone debris. Collision group is already DEBRIS.
  if (hasComponent(world, vehiclePartEid, CompoundPart)) {
    promoteCompoundPartToBody(vehiclePartEid);
    if (hasComponent(world, vehiclePartEid, VehiclePart)) {
      VehiclePart.removeComponent(world, vehiclePartEid);
    }
    return;
  }

  const jointPid = Joint.pid.get(vehiclePartEid);
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
  const turretEid = Tank.turretEId.get(vehicleEid);
  return getFilledSlotCount(vehicleEid) + getFilledSlotCount(turretEid);
}

export function getTankTotalSlotCount(vehicleEid: number, { world } = GameDI) {
  const { Tank } = getGameComponents(world);
  const turretEid = Tank.turretEId.get(vehicleEid);
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
  return TeamRef.id.get(tankEid);
}

export function getTankEngineLabel(vehicleEid: number, { world } = GameDI): string {
  const { Vehicle } = getGameComponents(world);
  const engine = Vehicle.engineType.get(vehicleEid) as EngineType;
  return EngineLabels[engine];
}
