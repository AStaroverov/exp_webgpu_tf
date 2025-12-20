import { addEntity, EntityId } from 'bitecs';
import { randomRangeFloat, randomRangeInt } from '../../../../../../../lib/random.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Children } from '../../Components/Children.ts';
import { Rock } from '../../Components/Rock.ts';
import {
    createRockParts,
    getRandomStoneColor,
} from './RockParts.ts';
import { generateGridRockShape, RockGridOptions } from './generateGridRockShape.ts';

/**
 * Options for creating a rock
 */
export type CreateRockOptions = {
    x: number;
    y: number;
    rotation?: number;
    color?: TColor;
    density?: number;
} & Partial<RockGridOptions>;

/**
 * Create a rock entity made of small destructible pieces.
 * Uses a grid-based approach with noise to create varied shapes.
 */
export function createRock(opts: CreateRockOptions, { world } = GameDI): EntityId {
    const cols = opts.cols ?? randomRangeInt(10, 30);
    const rows = opts.rows ?? randomRangeInt(10, 30);
    const cellSize = opts.cellSize ?? 6;
    const partSize = opts.partSize ?? 7;
    const rotation = opts.rotation ?? randomRangeFloat(0, Math.PI * 2);
    const color = opts.color ?? getRandomStoneColor();
    const noiseScale = opts.noiseScale ?? 0.05;
    const emptyThreshold = opts.emptyThreshold ?? 0.5;
    const density = opts.density ?? 1000;

    // Create rock entity (just a container for parts, no physics body)
    const rockEid = addEntity(world);
    Rock.addComponent(world, rockEid, 0);
    addTransformComponents(world, rockEid);
    Children.addComponent(world, rockEid);

    // Generate rock shape using grid + noise
    const partsData = generateGridRockShape({
        cols,
        rows,
        cellSize,
        partSize,
        noiseScale,
        noiseOctaves: 3,
        emptyThreshold,
    });

    // Create the rock parts
    createRockParts(rockEid, opts.x, opts.y, partsData, color, {
        rotation,
        density,
    });

    return rockEid;
}

export function createRockField(
    centerX: number,
    centerY: number,
    areaWidth: number,
    areaHeight: number,
    count: number,
): EntityId[] {
    const rocks: EntityId[] = [];

    for (let i = 0; i < count; i++) {
        const x = centerX + randomRangeFloat(-areaWidth / 2, areaWidth / 2);
        const y = centerY + randomRangeFloat(-areaHeight / 2, areaHeight / 2);
        rocks.push(createRock({ x, y }));
    }

    return rocks;
}
