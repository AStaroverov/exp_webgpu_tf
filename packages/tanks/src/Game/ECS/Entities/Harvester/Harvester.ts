import { PI } from '../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../Vehicle/VehicleParts.ts';
import { createTankExhaustPipes } from '../ExhaustPipe.ts';
import { createHarvesterBase, createHarvesterTracks, createHarvesterTurret } from './HarvesterBase.ts';
import {
    barrierSet,
    caterpillarLength,
    caterpillarSetLeft,
    caterpillarSetRight,
    DENSITY,
    hullSet,
    PADDING,
    PARTS_COUNT,
    scoopSet,
    shieldSet,
    SIZE,
    TRACK_ANCHOR_Y,
} from './HarvesterParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from './Options.ts';
import { EngineType } from '../../../Config/vehicles.ts';

// Harvester colors - industrial/utility look
const TRACKS_COLOR = new Float32Array([0.4, 0.4, 0.4, 1]);
const BARRIER_COLOR = new Float32Array([0.8, 0.6, 0.2, 1]); // Orange/yellow warning color
const SCOOP_COLOR = new Float32Array([0.6, 0.5, 0.3, 1]);   // Rusty metal color
const SHIELD_COLOR = new Float32Array([0.3, 0.7, 1.0, 0.6]); // Cyan semi-transparent energy shield
const APPROXIMATE_COLLIDER_RADIUS = 85;

export function createHarvester(opts: {
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
    options.vehicleType = VehicleType.Harvester;
    options.engineType = EngineType.v12;
    options.trackLength = caterpillarLength;

    // Heavy base for bulldozer
    options.density = DENSITY * 16;
    options.width = PADDING * 10;
    options.height = PADDING * 10;
    const [harvesterEid, harvesterPid] = createHarvesterBase(options);

    // Create left and right tracks as independent entities
    const [leftTrackEid, rightTrackEid] = createHarvesterTracks(
        options,
        {
            anchorX: 0,
            leftAnchorY: TRACK_ANCHOR_Y,
            rightAnchorY: -TRACK_ANCHOR_Y,
            trackWidth: PADDING * 2,
            trackHeight: options.trackLength,
        },
        harvesterEid,
        harvesterPid,
    );

    options.density = DENSITY * 2;
    options.width = PADDING * 10;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = PI * 0.4; // Slower rotation for heavy barrier
    const [barrierEid] = createHarvesterTurret(options, harvesterEid, harvesterPid);

    // Hull parts attached to harvester body
    createSlotEntities(harvesterEid, hullSet, options.color, SlotPartType.HullPart);
    
    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(leftTrackEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Front scoop for collecting debris
    updateColorOptions(options, SCOOP_COLOR);
    createSlotEntities(harvesterEid, scoopSet, options.color, SlotPartType.Scoop);

    // Impenetrable barrier on the "turret"
    updateColorOptions(options, BARRIER_COLOR);
    createSlotEntities(barrierEid, barrierSet, options.color, SlotPartType.Barrier);

    // Energy shield arc - semi-transparent cyan, bullet-only collision
    updateColorOptions(options, SHIELD_COLOR);
    createSlotEntities(barrierEid, shieldSet, options.color, SlotPartType.Shield);

    // Fill all slots with physical parts
    updateSlotsBrightness(harvesterEid);
    fillAllSlots(harvesterEid, options);
    updateSlotsBrightness(leftTrackEid);
    fillAllSlots(leftTrackEid, options);
    updateSlotsBrightness(rightTrackEid);
    fillAllSlots(rightTrackEid, options);
    updateSlotsBrightness(barrierEid);
    fillAllSlots(barrierEid, options);

    // Add exhaust pipes
    createTankExhaustPipes(harvesterEid, PADDING * 10, PADDING * 10);

    return harvesterEid;
}
