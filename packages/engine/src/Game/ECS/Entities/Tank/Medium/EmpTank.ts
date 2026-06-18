import { TColor } from "renderer/src/ECS/Components/Common.ts";
import { SlotPartType } from "../../../Components/SlotConfig.ts";
import { VehicleType } from "../../../Components/Vehicle.ts";
import {
  createSlotEntities,
  fillAllSlots,
  updateSlotsBrightness,
} from "../../Vehicle/VehicleParts.ts";
import { mutatedOptions, resetOptions, updateColorOptions } from "../Common/Options.ts";
import { createTankBase, createTankTracks, createTankTurret } from "../Common/Tank.ts";
import { createTankExhaustPipes } from "../../ExhaustPipe.ts";
import {
  caterpillarLength,
  caterpillarSetLeft,
  caterpillarSetRight,
  headlightSet,
  hullSet,
  PADDING,
  PARTS_COUNT,
  SIZE,
  TRACK_ANCHOR_Y,
  turretGunSet,
  turretHeadSet,
} from "./MediumTankParts.ts";
import { getTankConfig, HeadlightConfig } from "../../../../Config/vehicles.ts";
import { TurretSpeedConfig } from "../../../../Config/weapons.ts";
import { EmpVfxConfig } from "../../../../Config/vfx.ts";
import { randomVehiclePalette } from "../../../../Config/vehiclePalette.ts";

// Pale electric blue — marks the EMP turret head so the loadout reads at a glance.
const EMP_TURRET_ACCENT = new Float32Array([...EmpVfxConfig.tint, 1]);

/**
 * EMP tank on the Medium chassis: same hull/tracks/turret geometry as the
 * medium tank, but the gun fires the slow EMP grenade. A normal `Firearms`
 * turret — the caliber comes from the vehicle type's config row.
 */
export function createEmpTank(opts: {
  playerId: number;
  teamId: number;
  x: number;
  y: number;
  rotation: number;
  color: TColor;
}) {
  const config = getTankConfig(VehicleType.EmpTank);
  const gun = config.gun;
  if (gun === undefined) throw new Error(`Vehicle type ${VehicleType.EmpTank} has no gun armament`);

  const options = resetOptions(mutatedOptions, opts);
  options.partsCount = PARTS_COUNT;
  options.size = SIZE;
  options.padding = PADDING;
  options.vehicleType = VehicleType.EmpTank;
  options.engineType = config.engine;
  options.trackLength = caterpillarLength;

  options.density = config.density * 14;
  options.width = PADDING * 11;
  options.height = PADDING * 7;
  const [tankEid, tankPid] = createTankBase(options);

  // Create left and right tracks as independent entities
  const [leftTrackEid, rightTrackEid] = createTankTracks(
    options,
    {
      leftAnchorY: TRACK_ANCHOR_Y,
      rightAnchorY: -TRACK_ANCHOR_Y,
      anchorX: 0,
      trackWidth: PADDING * 2,
      trackHeight: caterpillarLength,
    },
    tankEid,
    tankPid,
  );

  options.density = config.density;
  options.width = PADDING * 6;
  options.height = PADDING * 5;
  options.turret.rotationSpeed = TurretSpeedConfig.medium;
  options.turret.gunWidth = PADDING * 6;
  options.turret.gunHeight = PADDING * 2;
  options.firearms.bulletCaliber = gun.caliber;
  options.spawnDeltaPosition = [11 * PADDING, 0];
  const [turretEid, gunEid] = createTankTurret(options, tankEid, tankPid);

  // Add exhaust pipes
  createTankExhaustPipes(tankEid, PADDING * 11, PADDING * 7);

  // Body uses a random contrastive palette; the gun carries the team color.
  const palette = randomVehiclePalette();

  // Hull parts attached to tank body
  updateColorOptions(options, palette.hull);
  createSlotEntities(tankEid, hullSet, options.color, SlotPartType.HullPart);
  createSlotEntities(tankEid, headlightSet, HeadlightConfig.color, SlotPartType.Headlight);

  // Caterpillar parts attached to track entities
  updateColorOptions(options, palette.tracks);
  createSlotEntities(leftTrackEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
  createSlotEntities(rightTrackEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

  // Gun (orudie) = team color; turret head gets the electric-blue EMP accent.
  updateColorOptions(options, opts.color);
  createSlotEntities(gunEid, turretGunSet, options.color, SlotPartType.TurretGun);
  updateColorOptions(options, EMP_TURRET_ACCENT);
  createSlotEntities(turretEid, turretHeadSet, options.color, SlotPartType.TurretHead);

  // Fill all slots with physical parts
  updateSlotsBrightness(tankEid);
  fillAllSlots(tankEid, options);
  updateSlotsBrightness(leftTrackEid);
  fillAllSlots(leftTrackEid, options);
  updateSlotsBrightness(rightTrackEid);
  fillAllSlots(rightTrackEid, options);
  updateSlotsBrightness(turretEid);
  fillAllSlots(turretEid, options);
  updateSlotsBrightness(gunEid);
  fillAllSlots(gunEid, options);

  return tankEid;
}
