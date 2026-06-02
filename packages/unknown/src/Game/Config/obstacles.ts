/**
 * Obstacles Configuration
 * 
 * Configuration for game obstacles: rocks, walls, and other static entities.
 */

// =============================================================================
// ROCK GENERATION
// =============================================================================

export const RockConfig = {
    /** Column count range [min, max] */
    colsRange: [7, 12] as [number, number],
    
    /** Row count range [min, max] */
    rowsRange: [7, 12] as [number, number],
    
    /** Cell size range [min, max] */
    cellSizeRange: [8, 12] as [number, number],
    
    /** Part size range [min, max] */
    partSizeRange: [5, 12] as [number, number],
    
    /** Noise scale range for procedural generation */
    noiseScaleRange: [0.03, 0.04] as [number, number],
    
    /** Noise octaves range */
    noiseOctavesRange: [1, 3] as [number, number],
    
    /** Empty threshold range (higher = more empty space) */
    emptyThresholdRange: [0.5, 0.8] as [number, number],
    
    /** Default rock density */
    defaultDensity: 1000,
} as const;

export type RockType = typeof RockConfig;

// =============================================================================
// BUILDING GENERATION (Abandoned/Ruined Buildings)
// =============================================================================

export const BuildingConfig = {
    /** Column count range [min, max] - number of rooms horizontally */
    colsRange: [4, 8] as [number, number],

    /** Row count range [min, max] - number of rooms vertically */
    rowsRange: [4, 8] as [number, number],

    /** Cell/room size range [min, max] */
    cellSizeRange: [20, 35] as [number, number],

    /** Wall thickness range [min, max] */
    wallThicknessRange: [8, 16] as [number, number],

    /** Noise scale range for procedural destruction */
    noiseScaleRange: [0.1, 0.2] as [number, number],

    /** Noise octaves range */
    noiseOctavesRange: [2, 4] as [number, number],

    /** Destruction threshold range (higher = more destruction) */
    destructionThresholdRange: [0.1, 0.3] as [number, number],

    /** Interior wall chance range */
    interiorWallChanceRange: [0.2, 0.5] as [number, number],

    /** Default building density */
    defaultDensity: 5000,
} as const;

export type BuildingType = typeof BuildingConfig;

// =============================================================================
// HEX-GRID PLACEMENT (prebuild → validate → commit, see spawnObstacles.ts)
// =============================================================================

export const ObstacleConfig = {
    /** Probability of placing a rock on a free cell during prebuild. */
    spawnChance: 0.12,
    /** How many times to re-roll the whole layout when validate() rejects it. */
    maxLayoutAttempts: 5,
} as const;

export type ObstacleConfigType = typeof ObstacleConfig;

