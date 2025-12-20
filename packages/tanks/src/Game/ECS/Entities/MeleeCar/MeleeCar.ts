import { PI } from '../../../../../../../lib/math.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from '../../Systems/Vehicle/VehicleControllerSystems.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../Vehicle/VehicleParts.ts';
import { addCarExhaustPipe } from '../ExhaustPipe.ts';
import { createMeleeCarBase, createMeleeCarWheels } from './MeleeCarBase.ts';
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
}) {
    const options = resetOptions(mutatedOptions, opts);
    options.partsCount = PARTS_COUNT;
    options.size = SIZE;
    options.padding = PADDING;
    options.approximateColliderRadius = APPROXIMATE_COLLIDER_RADIUS;
    options.vehicleType = VehicleType.MeleeCar;
    options.engineType = VehicleEngineType.v8; // Fast v8 engine
    options.wheelBase = wheelBase;

    // Light and fast base
    options.density = DENSITY * 8;
    options.width = PADDING * 6;
    options.height = PADDING * 10;
    options.linearDamping = 3; // Less friction than tanks
    options.angularDamping = 4; // Quick turning
    const [carEid, carPid] = createMeleeCarBase(options);

    // Create 4 wheels as independent entities
    // Front wheels are steerable, rear wheels are drive wheels
    const [frontLeftEid, frontRightEid, rearLeftEid, rearRightEid] = createMeleeCarWheels(
        options,
        {
            ...WHEEL_ANCHORS,
            wheelWidth: WHEEL_WIDTH,
            wheelHeight: WHEEL_HEIGHT,
            maxSteeringAngle: PI / 5, // ~36 degrees max steering
            steeringSpeed: PI * 3, // Fast steering response
        },
        carEid,
        carPid,
    );

    // Hull parts - main car body
    createSlotEntities(carEid, hullSet, options.color, SlotPartType.HullPart);

    // Wheel visual parts attached to each wheel entity
    updateColorOptions(options, WHEEL_COLOR);
    createSlotEntities(frontLeftEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(frontRightEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(rearLeftEid, wheelSlotSet, options.color, SlotPartType.Wheel);
    createSlotEntities(rearRightEid, wheelSlotSet, options.color, SlotPartType.Wheel);

    // Fill all slots with physical parts
    updateSlotsBrightness(carEid);
    fillAllSlots(carEid, options);
    updateSlotsBrightness(frontLeftEid);
    fillAllSlots(frontLeftEid, options);
    updateSlotsBrightness(frontRightEid);
    fillAllSlots(frontRightEid, options);
    updateSlotsBrightness(rearLeftEid);
    fillAllSlots(rearLeftEid, options);
    updateSlotsBrightness(rearRightEid);
    fillAllSlots(rearRightEid, options);

    // Add exhaust pipe
    addCarExhaustPipe(carEid, PADDING * 6, PADDING * 10);

    return carEid;
}

