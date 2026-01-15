import { addEntity } from 'bitecs';
import { randomRangeFloat, randomRangeInt } from '../../../../../../../lib/random.ts';
import { TColor } from '../../../../../../renderer/src/ECS/Components/Common.ts';
import { addTransformComponents } from '../../../../../../renderer/src/ECS/Components/Transform.ts';
import { GameDI } from '../../../DI/GameDI.ts';
import { Children } from '../../Components/Children.ts';
import { Building } from '../../Components/Building.ts';
import {
    createBuildingParts,
    BuildingMaterial,
} from './BuildingParts.ts';
import { generateBuildingShape, BuildingGridOptions } from './generateBuildingShape.ts';
import { BuildingConfig } from '../../../Config/index.ts';

/**
 * Options for creating an abandoned/ruined building
 */
export type CreateBuildingOptions = {
    x: number;
    y: number;
    rotation?: number;
    color?: TColor;
    density?: number;
    material?: BuildingMaterial;
} & Partial<BuildingGridOptions>;

/**
 * Create an abandoned/ruined building entity made of destructible wall pieces.
 * Uses a grid-based approach with noise to create varied destruction patterns.
 */
export function createBuilding(opts: CreateBuildingOptions, { world } = GameDI) {
    const cols = opts.cols ?? randomRangeInt(...BuildingConfig.colsRange);
    const rows = opts.rows ?? randomRangeInt(...BuildingConfig.rowsRange);
    const cellSize = opts.cellSize ?? randomRangeInt(...BuildingConfig.cellSizeRange);
    const wallThickness = opts.wallThickness ?? randomRangeInt(...BuildingConfig.wallThicknessRange);
    const rotation = opts.rotation ?? 0; // Buildings don't rotate, only individual debris pieces do
    const noiseScale = opts.noiseScale ?? randomRangeFloat(...BuildingConfig.noiseScaleRange);
    const noiseOctaves = opts.noiseOctaves ?? randomRangeInt(...BuildingConfig.noiseOctavesRange);
    const destructionThreshold = opts.destructionThreshold ?? randomRangeFloat(...BuildingConfig.destructionThresholdRange);
    const interiorWallChance = opts.interiorWallChance ?? randomRangeFloat(...BuildingConfig.interiorWallChanceRange);

    const material = opts.material ?? (Math.random() > 0.5 ? 'concrete' : 'brick');
    const density = opts.density ?? BuildingConfig.defaultDensity;

    // Generate building shape using grid + noise
    const partsData = generateBuildingShape({
        cols,
        rows,
        cellSize,
        wallThickness,
        noiseScale,
        noiseOctaves,
        destructionThreshold,
        interiorWallChance,
    });

    if (partsData.length === 0) return;

    // Create building entity (container for parts)
    const buildingEid = addEntity(world);
    Building.addComponent(world, buildingEid);
    addTransformComponents(world, buildingEid);
    Children.addComponent(world, buildingEid);

    // Create the building parts
    createBuildingParts(buildingEid, opts.x, opts.y, partsData, material, {
        rotation,
        density,
    });

    return buildingEid;
}

/**
 * Create a small ruined shack/hut
 */
export function createSmallRuin(x: number, y: number, rotation?: number) {
    return createBuilding({
        x,
        y,
        rotation,
        cols: randomRangeInt(4, 6),
        rows: randomRangeInt(4, 6),
        cellSize: randomRangeInt(15, 25),
        wallThickness: randomRangeInt(8, 12),
        destructionThreshold: 0.4 + randomRangeFloat(0.3, 0.5),
        interiorWallChance: 0.1,
    });
}

/**
 * Create a medium ruined building
 */
export function createMediumRuin(x: number, y: number, rotation?: number) {
    return createBuilding({
        x,
        y,
        rotation,
        cols: randomRangeInt(6, 8),
        rows: randomRangeInt(6, 8),
        cellSize: randomRangeInt(20, 30),
        wallThickness: randomRangeInt(12, 16),
        destructionThreshold: 0.4 + randomRangeFloat(0.25, 0.45),
        interiorWallChance: randomRangeFloat(0.2, 0.4),
    });
}

/**
 * Create a large ruined building/warehouse
 */
export function createLargeRuin(x: number, y: number, rotation?: number) {
    return createBuilding({
        x,
        y,
        rotation,
        cols: randomRangeInt(8, 10),
        rows: randomRangeInt(8, 10),
        cellSize: randomRangeInt(25, 40),
        wallThickness: randomRangeInt(16, 20),
        destructionThreshold: 0.4 + randomRangeFloat(0.2, 0.4),
        interiorWallChance: randomRangeFloat(0.3, 0.5),
        material: 'concrete',
    });
}

