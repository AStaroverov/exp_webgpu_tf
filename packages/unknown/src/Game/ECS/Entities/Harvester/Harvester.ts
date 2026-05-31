import { PhysicsWorld } from '../../createPhysicsWorld.ts';
import { PhysicalWorld } from '../../../Physical/initPhysicalWorld.ts';
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
import { Worlds } from '../../../DI/Worlds.ts';

// Harvester colors - industrial/utility look
const TRACKS_COLOR = new Float32Array([0.4, 0.4, 0.4, 1]);
const BARRIER_COLOR = new Float32Array([0.8, 0.6, 0.2, 1]); // Orange/yellow warning color
const SCOOP_COLOR = new Float32Array([0.6, 0.5, 0.3, 1]);   // Rusty metal color
const SHIELD_COLOR = new Float32Array([0.3, 0.7, 1.0, 0.6]); // Cyan semi-transparent energy shield
const APPROXIMATE_COLLIDER_RADIUS = 85;

export function createHarvester(world: PhysicsWorld, physicalWorld: PhysicalWorld, opts: {
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
    const renderWorld = Worlds.renderWorld;

    options.density = DENSITY * 16;
    options.width = PADDING * 10;
    options.height = PADDING * 10;
    const [harvesterPhysEid, harvesterRenderEid, harvesterPid] = createHarvesterBase(world, physicalWorld, options);

    // Create left and right tracks as independent entities
    const [leftTrackRenderEid, rightTrackRenderEid] = createHarvesterTracks(
        world,
        physicalWorld,
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
    const [barrierRenderEid] = createHarvesterTurret(world, physicalWorld, options, harvesterPhysEid, harvesterRenderEid, harvesterPid);

    // Hull parts attached to harvester body
    createSlotEntities(renderWorld, harvesterRenderEid, hullSet, options.color, SlotPartType.HullPart);

    // Caterpillar parts attached to track entities
    updateColorOptions(options, TRACKS_COLOR);
    createSlotEntities(renderWorld, leftTrackRenderEid, caterpillarSetLeft, options.color, SlotPartType.Caterpillar);
    createSlotEntities(renderWorld, rightTrackRenderEid, caterpillarSetRight, options.color, SlotPartType.Caterpillar);

    // Front scoop for collecting debris
    updateColorOptions(options, SCOOP_COLOR);
    createSlotEntities(renderWorld, harvesterRenderEid, scoopSet, options.color, SlotPartType.Scoop);

    // Impenetrable barrier on the "turret"
    updateColorOptions(options, BARRIER_COLOR);
    createSlotEntities(renderWorld, barrierRenderEid, barrierSet, options.color, SlotPartType.Barrier);

    // Energy shield arc - semi-transparent cyan, bullet-only collision
    updateColorOptions(options, SHIELD_COLOR);
    createSlotEntities(renderWorld, barrierRenderEid, shieldSet, options.color, SlotPartType.Shield);

    // Fill all slots with physical parts
    updateSlotsBrightness(renderWorld, harvesterRenderEid);
    fillAllSlots(renderWorld, physicalWorld, harvesterRenderEid, options);
    updateSlotsBrightness(renderWorld, leftTrackRenderEid);
    fillAllSlots(renderWorld, physicalWorld, leftTrackRenderEid, options);
    updateSlotsBrightness(renderWorld, rightTrackRenderEid);
    fillAllSlots(renderWorld, physicalWorld, rightTrackRenderEid, options);
    updateSlotsBrightness(renderWorld, barrierRenderEid);
    fillAllSlots(renderWorld, physicalWorld, barrierRenderEid, options);

    // Add exhaust pipes
    createTankExhaustPipes(renderWorld, harvesterRenderEid, PADDING * 10, PADDING * 10);

    return harvesterPhysEid;
}
