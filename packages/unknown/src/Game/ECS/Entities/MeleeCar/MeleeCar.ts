import { PI } from '../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { createSlotEntities, fillAllSlots } from '../Vehicle/VehicleParts.ts';
import { getPhysicsOf } from '../../refs.ts';
import { addCarExhaustPipe } from '../ExhaustPipe.ts';
import { createMeleeCarBase, createMeleeCarWheels } from './MeleeCarBase.ts';
import { DampingConfig, EngineType } from '../../../Config/index.ts';
import {
    DENSITY, hullSet,
    PADDING,
    PARTS_COUNT,
    SIZE,
    WHEEL_ANCHORS,
    WHEEL_HEIGHT,
    WHEEL_WIDTH,
    wheelBase,
    wheelSlotSet
} from './MeleeCarParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from './Options.ts';
import { Worlds } from '../../../DI/Worlds.ts';

// MeleeCar colors - aggressive sporty look
const WHEEL_COLOR = new Float32Array([0.2, 0.2, 0.2, 1]);     // Dark rubber
const APPROXIMATE_COLLIDER_RADIUS = 45;

export function createMeleeCar(opts: {
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
    options.vehicleType = VehicleType.MeleeCar;
    options.engineType = EngineType.v8; // Fast v8 engine
    options.wheelBase = wheelBase;

    // Light and fast base
    options.density = DENSITY * 8;
    options.width = PADDING * 10;
    options.height = PADDING * 6;
    options.linearDamping = DampingConfig.carLinear;
    options.angularDamping = DampingConfig.carAngular;
    const [carPhysEid, carRenderEid, carPid] = createMeleeCarBase(options);

    // Create 4 wheels as independent entities
    // Front wheels are steerable, rear wheels are drive wheels
    const [frontLeftRenderEid, frontRightRenderEid, rearLeftRenderEid, rearRightRenderEid] = createMeleeCarWheels(
        options,
        {
            ...WHEEL_ANCHORS,
            wheelWidth: WHEEL_WIDTH,
            wheelHeight: WHEEL_HEIGHT,
            maxSteeringAngle: PI / 5, // ~36 degrees max steering
            steeringSpeed: PI * 3, // Fast steering response
        },
        carRenderEid,
        carPid,
    );

    // Carrier PHYSICS atoms (the EXACT atom owning each slot group).
    const frontLeftPhysEid = getPhysicsOf(frontLeftRenderEid);
    const frontRightPhysEid = getPhysicsOf(frontRightRenderEid);
    const rearLeftPhysEid = getPhysicsOf(rearLeftRenderEid);
    const rearRightPhysEid = getPhysicsOf(rearRightRenderEid);

    // Hull parts - main car body
    createSlotEntities(carPhysEid, hullSet, options.color, SlotPartType.HullPart);

    // Wheel visual parts attached to each wheel entity
    updateColorOptions(options, WHEEL_COLOR);
    createSlotEntities(frontLeftPhysEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(frontRightPhysEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(rearLeftPhysEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(rearRightPhysEid, wheelSlotSet, options.color, SlotPartType.Wheel);

    // Fill all slots with physical parts
    fillAllSlots(carPhysEid, options);
    fillAllSlots(frontLeftPhysEid, options);
    fillAllSlots(frontRightPhysEid, options);
    fillAllSlots(rearLeftPhysEid, options);
    fillAllSlots(rearRightPhysEid, options);

    // Add exhaust pipe
    addCarExhaustPipe(renderWorld, carRenderEid, PADDING * 10, PADDING * 6);

    return carPhysEid;
}

