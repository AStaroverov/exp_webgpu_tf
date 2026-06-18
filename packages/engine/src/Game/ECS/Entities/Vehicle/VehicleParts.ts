import { TColor } from "renderer/src/ECS/Components/Common.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { GameDI } from "../../../DI/GameDI.ts";
import { createRectangle } from "renderer/src/ECS/Entities/Shapes.ts";
import {
  getMatrixTranslationX,
  getMatrixTranslationY,
  LocalTransform,
} from "renderer/src/ECS/Components/Transform.ts";
import { attachRigidRectangleCollider } from "../../../Physical/createRigid.ts";
import { defaultVehicleOptions, VehicleOptions } from "./Options.ts";
import { randomRangeFloat } from "../../../../../../../lib/random.ts";
import { clamp } from "lodash-es";
import { addEntity, type EntityId, hasComponent } from "bitecs";
import { cos, min, sin } from "../../../../../../../lib/math.ts";
import { VehicleType } from "../../Components/Vehicle.ts";
import { getSlotPartConfig, SlotPartType } from "../../Components/SlotConfig.ts";
import { isSlot, isSlotEmpty, isSlotFilled } from "../../Utils/SlotUtils.ts";
import { HeadlightConfig } from "../../../Config/vehicles.ts";

export type PartsData = [x: number, y: number, w: number, h: number];

export function createRectangleSet(
  cols: number,
  rows: number,
  width: number,
  paddingWidth: number,
  height = width,
  paddingHeight = paddingWidth,
): PartsData[] {
  const count = cols * rows;
  return Array.from({ length: count }, (_, i) => {
    return [
      ((i * paddingWidth) % (paddingWidth * cols)) - ((paddingWidth * cols) / 2 - width / 2),
      Math.floor(i / cols) * paddingHeight - ((paddingHeight * rows) / 2 - height / 2),
      width,
      height,
    ];
  });
}

/** Headlight slots just beyond the hull's front (+X) edge: two pairs (left and
 *  right side), gap in the middle — car-style. `hullCols`/`hullRows` are the
 *  hull set's grid dimensions. */
export function createHeadlightSet(
  hullCols: number,
  hullRows: number,
  size: number,
  padding: number,
): PartsData[] {
  const frontX = (padding * hullCols) / 2 + size / 2;
  const clusterY = (padding * hullRows) / 4; // halfway between center and hull edge
  return [-clusterY, clusterY].flatMap(
    (cy) =>
      [
        [frontX, cy - padding / 2, size, size],
        [frontX, cy + padding / 2, size, size],
      ] as PartsData[],
  );
}

export function updateSlotsBrightness(parentEId: EntityId, { world } = GameDI) {
  const { Children, Slot } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);

  for (let i = 0; i < childCount; i++) {
    const slotEid = Children.entitiesIds.get(parentEId, i);
    if (!isSlot(slotEid)) continue;
    // Headlights stay uniformly white — no armor-texture brightness jitter.
    if (Slot.partType[slotEid] === SlotPartType.Headlight) continue;
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
  const { Slot, Color, Children, Parent } = getGameComponents(world);

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

    Children.addChildren(parentEId, slotEid);
  }
}

export function fillAllSlots(
  parentEId: EntityId,
  options: VehicleOptions,
  { world } = GameDI,
): void {
  const { Children } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);

  for (let i = 0; i < childCount; i++) {
    const childEid = Children.entitiesIds.get(parentEId, i);
    if (isSlot(childEid) && isSlotEmpty(childEid)) {
      fillSlot(childEid, options);
    }
  }
}

const fillSlotOptions: VehicleOptions = structuredClone(defaultVehicleOptions);
export function fillSlot(slotEid: EntityId, options: VehicleOptions, { world } = GameDI) {
  const {
    Slot,
    Parent,
    Children,
    Vehicle,
    RigidBodyRef,
    Color,
    VehiclePart,
    PlayerRef,
    TeamRef,
    Hitable,
    Damagable,
    VehiclePartCaterpillar,
    LightEmitter,
    CompoundPart,
  } = getGameComponents(world);

  if (isNaN(options.x) || isNaN(options.y) || isNaN(options.rotation)) {
    throw new Error("Some options are not set");
  }
  if (isSlotFilled(slotEid)) {
    return;
  }
  Object.assign(fillSlotOptions, options);

  const vehicleOrTurretEid = Parent.id.get(slotEid);
  const vehicleEid = hasComponent(world, vehicleOrTurretEid, Vehicle)
    ? vehicleOrTurretEid
    : Parent.id.get(vehicleOrTurretEid);
  const vehicleType = Vehicle.type.get(vehicleEid) as VehicleType;

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

  // Resolve the body this collider attaches to. Usually the slot's parent
  // (hull/turret/track) owns the body. For a bodiless structural frame (the gun,
  // rigidly fixed to the turret with no relative rotation) walk up to the nearest
  // body-owning ancestor and fold the frame's local offset into the anchor.
  let ownerEid = vehicleOrTurretEid;
  let ownerAnchorX = anchorX;
  let ownerAnchorY = anchorY;
  while (!hasComponent(world, ownerEid, RigidBodyRef)) {
    const frameLocal = LocalTransform.matrix.getBatch(ownerEid);
    ownerAnchorX += getMatrixTranslationX(frameLocal);
    ownerAnchorY += getMatrixTranslationY(frameLocal);
    ownerEid = Parent.id.get(ownerEid);
  }
  const rbId = RigidBodyRef.id[ownerEid];

  // Render-only entity + a collider on the owner body at the (composed) anchor.
  // No body, no joint. CompoundPart carries the owner+anchor (so the transform
  // system can place it) and the collider handle (so contact drains attribute
  // hits to this part). The owner frame already rotates, so the offset is the
  // unrotated anchor.
  const eid = createRectangle(world, fillSlotOptions);
  const colliderHandle = attachRigidRectangleCollider(rbId, {
    ...fillSlotOptions,
    offsetX: ownerAnchorX,
    offsetY: ownerAnchorY,
  });
  VehiclePart.addComponent(world, eid);
  CompoundPart.addComponent(world, eid, ownerEid, colliderHandle, ownerAnchorX, ownerAnchorY);

  PlayerRef.addComponent(world, eid, fillSlotOptions.playerId);
  TeamRef.addComponent(world, eid, fillSlotOptions.teamId);
  Hitable.addComponent(world, eid, min(fillSlotOptions.width, fillSlotOptions.height));
  Damagable.addComponent(world, eid, min(fillSlotOptions.width, fillSlotOptions.height) / 20);

  Parent.addComponent(world, eid, slotEid);
  Children.addChildren(slotEid, eid);

  if (partType === SlotPartType.Caterpillar) {
    VehiclePartCaterpillar.addComponent(world, eid);
  }

  if (partType === SlotPartType.Headlight) {
    LightEmitter.addComponent(
      world,
      eid,
      HeadlightConfig.directional ? -HeadlightConfig.intensity : HeadlightConfig.intensity,
    );
  }
}

export function getEmptySlotsCount(parentEId: EntityId, { world } = GameDI): number {
  const { Children } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);
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

export function findFirstEmptySlot(parentEId: EntityId, { world } = GameDI): EntityId | null {
  const { Children } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);
  for (let i = 0; i < childCount; i++) {
    const childEid = Children.entitiesIds.get(parentEId, i);
    if (isSlot(childEid) && isSlotEmpty(childEid)) {
      return childEid;
    }
  }
  return null;
}

export function getSlotCount(parentEId: EntityId, { world } = GameDI): number {
  const { Children } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);
  let count = 0;
  for (let i = 0; i < childCount; i++) {
    const childEid = Children.entitiesIds.get(parentEId, i);
    if (isSlot(childEid)) count++;
  }
  return count;
}

export function getFilledSlotCount(parentEId: EntityId, { world } = GameDI): number {
  const { Children } = getGameComponents(world);
  const childCount = Children.entitiesCount.get(parentEId);
  let filled = 0;
  for (let i = 0; i < childCount; i++) {
    const childEid = Children.entitiesIds.get(parentEId, i);
    if (isSlot(childEid) && isSlotFilled(childEid)) filled++;
  }
  return filled;
}

export function getVehicleTotalSlotCount(vehicleEid: EntityId, { world } = GameDI): number {
  const { Tank } = getGameComponents(world);
  const turretEid = Tank.turretEId.get(vehicleEid);
  return getSlotCount(vehicleEid) + getSlotCount(turretEid);
}

export function getVehicleFilledSlotCount(vehicleEid: EntityId, { world } = GameDI): number {
  const { Tank } = getGameComponents(world);
  const turretEid = Tank.turretEId.get(vehicleEid);
  return getFilledSlotCount(vehicleEid) + getFilledSlotCount(turretEid);
}

function adjustBrightness(eid: EntityId, start: number, end: number, { world } = GameDI) {
  const { Color } = getGameComponents(world);
  const factor = -1 * randomRangeFloat(start, end);
  Color.set$(
    eid,
    clamp(Color.getR(eid) + factor, 0, 1),
    clamp(Color.getG(eid) + factor, 0, 1),
    clamp(Color.getB(eid) + factor, 0, 1),
    Color.getA(eid),
  );
}
