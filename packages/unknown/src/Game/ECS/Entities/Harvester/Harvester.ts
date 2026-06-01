import { PI } from '../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots } from '../Vehicle/VehicleParts.ts';
import { getPhysicsOf } from '../../refs.ts';
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
import { Worlds } from '../../../DI/Worlds.ts';

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
}, { renderWorld } = Worlds) {
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
    const [harvesterPhysEid, harvesterRenderEid, harvesterPid] = createHarvesterBase(options);

    // Create left and right tracks as independent entities
    const [leftTrackRenderEid, rightTrackRenderEid] = createHarvesterTracks(
        options,
        {
            anchorX: 0,
            leftAnchorY: TRACK_ANCHOR_Y,
            rightAnchorY: -TRACK_ANCHOR_Y,
            trackWidth: PADDING * 2,
            trackHeight: options.trackLength,
        },
        harvesterRenderEid,
        harvesterPid,
    );

    options.density = DENSITY * 2;
    options.width = PADDING * 10;
    options.height = PADDING * 6;
    options.turret.rotationSpeed = PI * 0.4; // Slower rotation for heavy barrier
    const [barrierRenderEid] = createHarvesterTurret(options, harvesterPhysEid, harvesterRenderEid, harvesterPid);

    // Carrier PHYSICS atoms (the EXACT atom owning each slot group).
    const leftTrackPhysEid = getPhysicsOf(leftTrackRenderEid);
    const rightTrackPhysEid = getPhysicsOf(rightTrackRenderEid);
    const barrierPhysEid = getPhysicsOf(barrierRenderEid);

    // Hull parts attached to harvester body
    createSlotEntities(harvesterPhysEid, hullSet, options.color, SlotPartType.HullPart);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(leftTrackPhysEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(rightTrackPhysEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Front scoop for collecting debris
    updateColorOptions(options, SCOOP_COLOR);
    createSlotEntities(harvesterPhysEid, scoopSet, options.color, SlotPartType.Scoop);

    // Impenetrable barrier on the "turret"
    updateColorOptions(options, BARRIER_COLOR);
    createSlotEntities(barrierPhysEid, barrierSet, options.color, SlotPartType.Barrier);

    // Energy shield arc - semi-transparent cyan, bullet-only collision
    updateColorOptions(options, SHIELD_COLOR);
    createSlotEntities(barrierPhysEid, shieldSet, options.color, SlotPartType.Shield);

    // Fill all slots with physical parts
    fillAllSlots(harvesterPhysEid, options);
    fillAllSlots(leftTrackPhysEid, options);
    fillAllSlots(rightTrackPhysEid, options);
    fillAllSlots(barrierPhysEid, options);

    // Add exhaust pipes
    createTankExhaustPipes(renderWorld, harvesterRenderEid, PADDING * 10, PADDING * 10);

    return harvesterPhysEid;
}
