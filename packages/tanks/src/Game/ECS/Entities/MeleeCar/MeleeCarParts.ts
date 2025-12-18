import { createRectangleSet, PartsData } from '../Vehicle/VehicleParts.ts';

export const DENSITY = 200;
export const SIZE = 4;
export const PADDING = SIZE + 1;

// Compact rectangular hull - like a car body
export const hullSet = createRectangleSet(6, 10, SIZE, PADDING);

// Wheel configuration - 4 wheels at the corners
export const WHEEL_SIZE = 6;
export const WHEEL_PADDING = WHEEL_SIZE + 1;

// Create wheel positions at corners of the car
function createWheelSet(): PartsData[] {
    const result: PartsData[] = [];
    
    const hullHalfWidth = (6 * PADDING) / 2 + WHEEL_SIZE * 0.5;
    const frontY = -(10 * PADDING) / 2 + WHEEL_SIZE * 1.5;
    const rearY = (10 * PADDING) / 2 - WHEEL_SIZE * 1.5;
    
    // Front left wheel
    result.push([-hullHalfWidth, frontY, WHEEL_SIZE, WHEEL_SIZE * 1.5]);
    // Front right wheel
    result.push([hullHalfWidth, frontY, WHEEL_SIZE, WHEEL_SIZE * 1.5]);
    // Rear left wheel
    result.push([-hullHalfWidth, rearY, WHEEL_SIZE, WHEEL_SIZE * 1.5]);
    // Rear right wheel
    result.push([hullHalfWidth, rearY, WHEEL_SIZE, WHEEL_SIZE * 1.5]);
    
    return result;
}

export const wheelSet = createWheelSet();

// Calculate wheel base (distance between front and rear axle centers)
export const wheelBase = (10 * PADDING) - WHEEL_SIZE * 3;

export const PARTS_COUNT = hullSet.length + wheelSet.length;

