import { cos, floor, PI, sin } from '../../../../../../../lib/math.ts';
import { createRectangleSet, PartsData } from '../Vehicle/VehicleParts.ts';

export const DENSITY = 350;
export const SIZE = 6;
export const PADDING = SIZE + 1;

/**
 * Creates an arc of small rectangular parts for shield
 * @param count - number of parts in the arc
 * @param radius - radius of the arc from center
 * @param arcAngle - total angle of the arc in radians (centered at 0)
 * @param partWidth - width of each part
 * @param partHeight - height of each part
 */
export function createArcSet(
    count: number,
    radius: number,
    arcAngle: number,
    partWidth: number,
    partHeight: number,
): PartsData[] {
    const result: PartsData[] = [];
    const startAngle = -arcAngle / 2;
    const angleStep = arcAngle / (count - 1);

    for (let i = 0; i < count; i++) {
        const angle = startAngle + i * angleStep;
        // Arc faces forward (negative Y direction)
        const x = sin(angle) * radius;
        const y = -cos(angle) * radius;
        result.push([x, y, partWidth, partHeight]);
    }

    return result;
}

// Hull - wider and slightly shorter than medium tank (bulldozer shape)
export const hullSet = createRectangleSet(10, 12, SIZE, PADDING);

// Barrier "turret" - wide shield that rotates, no gun barrel
// Acts as impenetrable barrier instead of weapon
export const barrierSet = createRectangleSet(8, 4, SIZE, PADDING);

/**
 * Creates a U-shaped scoop (square without one side - open at the front for collecting)
 * @param sideLength - number of parts on each side arm
 * @param backLength - number of parts on the back wall
 * @param partSize - size of each part
 * @param partPadding - padding between parts
 * @param offsetY - vertical offset from center (negative = forward)
 */
export function createUScoopSet(
    sideLength: number,
    backLength: number,
    partSize: number,
    partPadding: number,
    offsetY: number,
): PartsData[] {
    const result: PartsData[] = [];
    
    const halfWidth = (backLength * partPadding) / 2 - partSize / 2;
    const backY = offsetY + (sideLength - 1) * partPadding; // Back wall position
    
    // Left side (vertical line, going from front to back)
    for (let i = 0; i < sideLength; i++) {
        const x = -halfWidth;
        const y = offsetY + i * partPadding;
        result.push([x, y, partSize, partSize]);
    }
    
    // Right side (vertical line, going from front to back)
    for (let i = 0; i < sideLength; i++) {
        const x = halfWidth;
        const y = offsetY + i * partPadding;
        result.push([x, y, partSize, partSize]);
    }
    
    // Back wall (horizontal line at the back, excluding corners already placed)
    for (let i = 1; i < backLength - 1; i++) {
        const x = i * partPadding - halfWidth;
        const y = backY;
        result.push([x, y, partSize, partSize]);
    }
    
    return result;
}

// Front scoop - U-shaped for collecting debris (square without back side)
// Positioned at the front of the harvester
export const SCOOP_SIDE_LENGTH = 6;  // Length of each side arm
export const SCOOP_FRONT_LENGTH = 10; // Width of the front
export const scoopSet: PartsData[] = createUScoopSet(
    SCOOP_SIDE_LENGTH,
    SCOOP_FRONT_LENGTH,
    SIZE,
    PADDING,
    -(PADDING * 10 + 3), // Offset forward
);

// Caterpillar configuration - heavy duty tracks
export const CATERPILLAR_SIZE = 4;
export const CATERPILLAR_PADDING = CATERPILLAR_SIZE + 1;
export const CATERPILLAR_LINE_COUNT = 20;
export const caterpillarLength = CATERPILLAR_LINE_COUNT * CATERPILLAR_PADDING;
export const caterpillarSet = createRectangleSet(
    2, CATERPILLAR_LINE_COUNT,
    SIZE, PADDING,
    CATERPILLAR_SIZE, CATERPILLAR_PADDING,
);
export const caterpillarSetLeft: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] += PADDING * 6 + SIZE * 0.3;
    return set;
});
export const caterpillarSetRight: PartsData[] = caterpillarSet.map((set) => {
    set = set.slice() as PartsData;
    set[0] -= PADDING * 6 + SIZE * 0.3;
    return set;
});

// Shield arc - energy barrier in front of the turret
// Arc of 48 densely packed parts, ~126 degree angle, 50% larger radius
export const SHIELD_PART_SIZE = 8;
export const SHIELD_RADIUS = SHIELD_PART_SIZE * 16;
export const SHIELD_ARC_ANGLE = PI * 0.6; // 108 degrees
export const SHIELD_PARTS_COUNT = 35; 
export const shieldSet = [
    ...createArcSet(
        SHIELD_PARTS_COUNT,
        SHIELD_RADIUS,
        SHIELD_ARC_ANGLE,
        SHIELD_PART_SIZE,
        SHIELD_PART_SIZE,
    ),
    ...createArcSet(
        floor(SHIELD_PARTS_COUNT*0.8),
        floor(SHIELD_RADIUS*0.8),
        SHIELD_ARC_ANGLE,
        floor(SHIELD_PART_SIZE*0.8),
        floor(SHIELD_PART_SIZE*0.8),
    )
]

export const PARTS_COUNT = hullSet.length + barrierSet.length + scoopSet.length + shieldSet.length + CATERPILLAR_LINE_COUNT * 2 * 2;
