import { PI } from '../../../../../../../../lib/math.ts';
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
} from '../Heavy/HeavyTankParts.ts';
import { EngineType, HeadlightConfig, ReloadConfig } from '../../../../Config/index.ts';

const TRACKS_COLOR = new Float32Array([0.5, 0.5, 0.5, 1]);
// Distinct dark red/orange turret so the launcher reads as a launcher.
const TURRET_COLOR = new Float32Array([0.8, 0.3, 0.2, 1]);
const APPROXIMATE_COLLIDER_RADIUS = 80;

export function createRocketTank(opts: {
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
    options.vehicleType = VehicleType.RocketTank;
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
    options.turret.rotationSpeed = PI * 0.08;
    options.turret.gunWidth = PADDING * 8;
    options.turret.gunHeight = PADDING * 2.2;
    options.firearms.reloadingDuration = ReloadConfig.rocketLauncher;
    options.firearms.bulletCaliber = BulletCaliber.Rocket;
    options.firearms.bulletStartPosition = [13 * PADDING, 0];
    const [turretEid, gunEid] = createTankTurret(options, tankEid, tankPid);

    // Hull parts attached to tank body
    createSlotEntities(tankEid, hullSet, options.color, SlotPartType.HullPart);
    createSlotEntities(tankEid, headlightSet, HeadlightConfig.color, SlotPartType.Headlight);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(leftTrackEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Turret parts
    updateColorOptions(options, TURRET_COLOR);
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
