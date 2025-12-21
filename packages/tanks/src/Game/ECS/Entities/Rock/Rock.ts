import { addEntity } from 'bitecs';
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
export function createRock(opts: CreateRockOptions, { world } = GameDI) {
    const cols = opts.cols ?? randomRangeInt(5, 20);
    const rows = opts.rows ?? randomRangeInt(5, 20);
    const cellSize = opts.cellSize ?? randomRangeInt(5, 10);
    const partSize = opts.partSize ?? randomRangeInt(5, 10);
    const rotation = opts.rotation ?? randomRangeFloat(0, Math.PI * 2);
    const noiseScale = opts.noiseScale ?? randomRangeFloat(0.03, 0.08);
    const noiseOctaves = opts.noiseOctaves ?? randomRangeInt(1, 5);
    const emptyThreshold = opts.emptyThreshold ?? randomRangeFloat(0.5, 0.8);
    
    const color = opts.color ?? getRandomStoneColor();
    const density = opts.density ?? 1000;

    // Generate rock shape using grid + noise
    const partsData = generateGridRockShape({
        cols,
        rows,
        cellSize,
        partSize,
        noiseScale,
        noiseOctaves,
        emptyThreshold,
    });

    if (partsData.length === 0) return;

    // Create rock entity (just a container for parts, no physics body)
    const rockEid = addEntity(world);
    Rock.addComponent(world, rockEid, 0);
    addTransformComponents(world, rockEid);
    Children.addComponent(world, rockEid);

    // Create the rock parts
    createRockParts(rockEid, opts.x, opts.y, partsData, color, {
        rotation,
        density,
    });
}
