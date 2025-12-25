import { createRectangleSet, PartsData } from '../Vehicle/VehicleParts.ts';

export const DENSITY = 200;
export const SIZE = 4;
export const PADDING = SIZE + 1;

// Compact rectangular hull - like a car body
export const hullSet = createRectangleSet(10, 6, SIZE, PADDING);

// Wheel configuration - 4 wheels at the corners
export const WHEEL_SIZE = 6;
export const WHEEL_PADDING = WHEEL_SIZE + 1;

// Hull dimensions for wheel positioning
const HULL_HALF_HEIGHT = (6 * PADDING) / 2 + WHEEL_SIZE * 0.5;
const FRONT_X = (10 * PADDING) / 2 - WHEEL_SIZE * 1.5;
const REAR_X = -(10 * PADDING) / 2 + WHEEL_SIZE * 1.5;

// Wheel anchor positions (x, y from car center)
export const WHEEL_ANCHORS = {
    frontLeft: { x: FRONT_X, y: -HULL_HALF_HEIGHT },
    frontRight: { x: FRONT_X, y: HULL_HALF_HEIGHT },
    rearLeft: { x: REAR_X, y: -HULL_HALF_HEIGHT },
    rearRight: { x: REAR_X, y: HULL_HALF_HEIGHT },
};

// Wheel visual dimensions
export const WHEEL_WIDTH = WHEEL_SIZE;
export const WHEEL_HEIGHT = WHEEL_SIZE * 1.5;

// Create wheel slot positions (for visual decoration attached to wheel entities)
function createWheelSet(): PartsData[] {
    // Each wheel will have its own slots for visual parts
    // These are in local coordinates relative to each wheel
    return [
        [0, 0, WHEEL_WIDTH, WHEEL_HEIGHT], // Local to wheel
    ];
}

export const wheelSlotSet = createWheelSet();

// Calculate wheel base (distance between front and rear axle centers)
export const wheelBase = FRONT_X - REAR_X;

export const PARTS_COUNT = hullSet.length + 4; // 4 wheels

