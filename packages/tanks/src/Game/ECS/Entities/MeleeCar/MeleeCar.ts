import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { SlotPartType } from '../../Components/SlotConfig.ts';
import { VehicleType } from '../../Components/Vehicle.ts';
import { VehicleEngineType } from '../../Systems/Vehicle/VehicleControllerSystems.ts';
import { createSlotEntities, fillAllSlots, updateSlotsBrightness } from '../Vehicle/VehicleParts.ts';
import { createMeleeCarBase } from './MeleeCarBase.ts';
import {
    DENSITY, detailSet, hullSet,
    PADDING,
    PARTS_COUNT,
    SIZE,
    wheelBase,
    wheelSet
} from './MeleeCarParts.ts';
import { mutatedOptions, resetOptions, updateColorOptions } from './Options.ts';

// MeleeCar colors - aggressive sporty look
const WHEEL_COLOR = new Float32Array([0.2, 0.2, 0.2, 1]);     // Dark rubber
const DETAIL_COLOR = new Float32Array([1.0, 0.3, 0.1, 1]);    // Orange/red accents
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
    const [carEid] = createMeleeCarBase(options);

    // Hull parts - main car body
    createSlotEntities(carEid, hullSet, options.color, SlotPartType.HullPart);

    // 4 wheels at corners
    updateColorOptions(options, WHEEL_COLOR);
    createSlotEntities(carEid, wheelSet, options.color, SlotPartType.Wheel);

    // Decorative details
    updateColorOptions(options, DETAIL_COLOR);
    createSlotEntities(carEid, detailSet, options.color, SlotPartType.Detail);

    updateSlotsBrightness(carEid);
    fillAllSlots(carEid, options);

    return carEid;
}

