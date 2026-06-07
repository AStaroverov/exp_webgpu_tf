import { TColor } from '../../../../../../../renderer/src/ECS/Components/Common.ts';

import { EntityId } from 'bitecs';
import { PI } from '../../../../../../../../lib/math.ts';
import { GameDI } from '../../../../DI/GameDI.ts';
import { getGameComponents } from '../../../createGameWorld.ts';
import { SlotPartType } from '../../../Components/SlotConfig.ts';
import { VehicleType } from '../../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../../Vehicle/VehicleParts.ts';
import { createVehicleTurret } from '../../Vehicle/VehicleBase.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from '../Common/Options.ts';
import { createTankBase, createTankTracks } from '../Common/Tank.ts';
import { createTankExhaustPipes } from '../../ExhaustPipe.ts';
import { createTurretHeadlight } from '../../TurretHeadlight.ts';
import { SIZE } from '../Medium/MediumTankParts.ts';
import {
    CATERPILLAR_LINE_COUNT,
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    headlightSet,
    hullSet,
    PADDING,
    TRACK_ANCHOR_Y,
    turretHeadSet,
} from '../Light/LightTankParts.ts';
import { EngineType, HeadlightConfig, SpotlightConfig } from '../../../../Config/vehicles.ts';

const TRACKS_COLOR = new Float32Array([0.6, 0.6, 0.6, 1]);
const TURRET_COLOR = new Float32Array([0.6, 1, 0.6, 1]);
const APPROXIMATE_COLLIDER_RADIUS = 50;

// No gun parts — the searchlight beam replaces the turret gun.
const PARTS_COUNT = hullSet.length + headlightSet.length + turretHeadSet.length + CATERPILLAR_LINE_COUNT * 2;

export function createRanger(opts: {
    playerId: number,
    teamId: number,
    x: number,
    y: number,
    rotation: number,
    color: TColor,
}, { world } = GameDI): EntityId {
    const { Tank } = getGameComponents(world);

    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.vehicleType = VehicleType.Ranger;
    options.engineType = EngineType.v6;
    options.trackLength = caterpillarLength;

    options.density = DENSITY * 14;
    options.width = PADDING * 10;
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

    // Turret without a gun: the searchlight beam takes the gun's place.
    options.density = DENSITY;
    options.width = PADDING * 6;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = PI * 0.3;
    const [turretEid] = createVehicleTurret(options, options.turret, tankEid, tankPid);
    Tank.setTurretEid(tankEid, turretEid);

    // Hull parts attached to tank body
    createSlotEntities(tankEid, hullSet, options.color, SlotPartType.HullPart);
    createSlotEntities(tankEid, headlightSet, HeadlightConfig.color, SlotPartType.Headlight);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(leftTrackEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Turret parts (the searchlight housing)
    updateColorOptions(options, TURRET_COLOR);
    createSlotEntities(turretEid, turretHeadSet, options.color, SlotPartType.TurretHead);

    // Fill all slots with physical parts
    updateSlotsBrightness(tankEid);
    fillAllSlots(tankEid, options);
    updateSlotsBrightness(leftTrackEid);
    fillAllSlots(leftTrackEid, options);
    updateSlotsBrightness(rightTrackEid);
    fillAllSlots(rightTrackEid, options);

    // Searchlight beam attached to the turret (directional light emitter).
    createTurretHeadlight(turretEid, {
        startX: PADDING * 2,
        startY: 0,
        length: PADDING * 40,
        nearWidth: PADDING * 3,
        farWidth: PADDING * 8,
        light: SpotlightConfig,
    });

    updateSlotsBrightness(turretEid);
    fillAllSlots(turretEid, options);

    // Add exhaust pipes
    createTankExhaustPipes(tankEid, PADDING * 10, PADDING * 8);

    return tankEid;
}
