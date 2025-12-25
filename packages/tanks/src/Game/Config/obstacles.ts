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
    colsRange: [5, 20] as [number, number],
    
    /** Row count range [min, max] */
    rowsRange: [5, 20] as [number, number],
    
    /** Cell size range [min, max] */
    cellSizeRange: [5, 10] as [number, number],
    
    /** Part size range [min, max] */
    partSizeRange: [5, 10] as [number, number],
    
    /** Noise scale range for procedural generation */
    noiseScaleRange: [0.03, 0.08] as [number, number],
    
    /** Noise octaves range */
    noiseOctavesRange: [1, 5] as [number, number],
    
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
    colsRange: [4, 10] as [number, number],

    /** Row count range [min, max] - number of rooms vertically */
    rowsRange: [4, 10] as [number, number],

    /** Cell/room size range [min, max] */
    cellSizeRange: [20, 40] as [number, number],

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
    defaultDensity: 1200,
} as const;

export type BuildingType = typeof BuildingConfig;

