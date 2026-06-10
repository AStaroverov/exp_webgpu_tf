import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';
import { BulletCaliber } from '../../../Components/Bullet.ts';
import { SlotPartType } from '../../../Components/SlotConfig.ts';
import { VehicleType } from '../../../Components/Vehicle.ts';
import { getGameComponents } from '../../../createGameWorld.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../../Vehicle/VehicleParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTracks, createTankTurret } from '../Common/Tank.ts';
import { createTankExhaustPipes } from '../../ExhaustPipe.ts';
import {
    cabinSet,
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    headlightSet,
    HULL_ROWS,
    hullSet,
    PADDING,
    PARTS_COUNT,
    railSet,
    RAIL_Y,
    SIZE,
    TRACK_ANCHOR_Y,
} from './RocketTankParts.ts';
import { HeadlightConfig, randomVehiclePalette, RocketTankConfig } from '../../../../Config/index.ts';

export function createRocketTank(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}) {
    const { world } = GameDI;
    const { HullAimed } = getGameComponents(world);

    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.vehicleType = VehicleType.RocketTank;
    options.engineType = RocketTankConfig.engine;
    options.trackLength = caterpillarLength;

    // Elongated hull — length (+X / forward) clearly exceeds width.
    options.density = RocketTankConfig.density * 14;
    options.width = PADDING * 14;
    options.height = PADDING * HULL_ROWS;
    const [tankEid, tankPid] = createTankBase(options);

    // The launcher is bolted to the hull, so the whole vehicle turns to aim.
    HullAimed.addComponent(world, tankEid);

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

    options.density = RocketTankConfig.density;
    options.width = PADDING * 16;
    options.height = PADDING * HULL_ROWS;
    options.turret.gunWidth = PADDING;
    options.turret.gunHeight = PADDING;
    options.firearms.bulletCaliber = BulletCaliber.Rocket;
    options.spawnDeltaPosition = [PADDING * 10, RAIL_Y];
    const [turretEid] = createTankTurret(options, tankEid, tankPid);

    // Body uses a random contrastive palette; the launch rail (weapon) = team color.
    const palette = randomVehiclePalette();

    // Hull parts attached to tank body
    updateColorOptions(options, palette.hull);
    createSlotEntities(tankEid, hullSet, options.color, SlotPartType.HullPart);
    createSlotEntities(tankEid, headlightSet, HeadlightConfig.color, SlotPartType.Headlight);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, palette.tracks);
    createSlotEntities(leftTrackEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Launch rail (orudie) = team color; pilot cabin from the palette — both
    // bolted to the (fixed) launcher carrier.
    updateColorOptions(options, opts.color);
    createSlotEntities(turretEid, railSet, options.color, SlotPartType.TurretGun);
    updateColorOptions(options, palette.turret);
    createSlotEntities(turretEid, cabinSet, options.color, SlotPartType.TurretHead);

    // Fill all slots with physical parts
    updateSlotsBrightness(tankEid);
    fillAllSlots(tankEid, options);
    updateSlotsBrightness(leftTrackEid);
    fillAllSlots(leftTrackEid, options);
    updateSlotsBrightness(rightTrackEid);
    fillAllSlots(rightTrackEid, options);
    updateSlotsBrightness(turretEid);
    fillAllSlots(turretEid, options);

    // Add exhaust pipes
    createTankExhaustPipes(tankEid, PADDING * 14, PADDING * HULL_ROWS);

    return tankEid;
}
