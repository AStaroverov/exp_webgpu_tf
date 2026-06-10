import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { SlotPartType } from '../../../Components/SlotConfig.ts';
import { VehicleType } from '../../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../../Vehicle/VehicleParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTracks, createTankTurret } from '../Common/Tank.ts';
import { createTankExhaustPipes } from '../../ExhaustPipe.ts';
import {
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    headlightSet,
    hullSet,
    PADDING,
    PARTS_COUNT,
    SIZE,
    TRACK_ANCHOR_Y,
    turretGunSet,
    turretHeadSet,
} from './HeavyTankParts.ts';
import { EngineType, HeadlightConfig, } from '../../../../Config/vehicles.ts';
import { TurretSpeedConfig } from '../../../../Config/weapons.ts';
import { randomVehiclePalette } from '../../../../Config/vehiclePalette.ts';

const APPROXIMATE_COLLIDER_RADIUS = 80;

export function createHeavyTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.vehicleType = VehicleType.HeavyTank;
    options.engineType = EngineType.v12;
    options.trackLength = caterpillarLength;

    options.density = DENSITY * 14;
    options.width = PADDING * 12;
    options.height = PADDING * 8;
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

    options.density = DENSITY;
    options.width = PADDING * 7;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = TurretSpeedConfig.heavy;
    options.turret.gunWidth = PADDING * 7;
    options.turret.gunHeight = PADDING * 2;
    options.firearms.bulletCaliber = BulletCaliber.Heavy;
    options.spawnDeltaPosition = [13 * PADDING, 0];
    const [turretEid, gunEid] = createTankTurret(options, tankEid, tankPid);

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

    // Turret + gun (orudie) = team color.
    updateColorOptions(options, opts.color);
    createSlotEntities(gunEid, turretGunSet, options.color, SlotPartType.TurretGun);
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

    // Add exhaust pipes
    createTankExhaustPipes(tankEid, PADDING * 12, PADDING * 8);

    return tankEid;
}
