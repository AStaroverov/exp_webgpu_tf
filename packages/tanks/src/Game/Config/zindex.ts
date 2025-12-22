/**
 * Z-Index Configuration for Rendering Order
 * 
 * All Z-index values are baseline/minimum values for entities.
 * Z-index is used to sort entities for rendering and shadow mapping.
 * Lower values render first (behind), higher values render on top.
 */

export const ZIndexConfig = {
    /** Background layer - always at the bottom */
    Background: 0,
    
    /** Tread marks left by tanks on the ground */
    TreadMark: 0.001,
    
    /** Rocks and terrain obstacles - related to terrain/size */
    Rock: 1,
    
    /** Spice resources on the ground */
    Spice: 0,
    
    /** Tank hull body */
    TankHull: 4,
    
    /** Tank caterpillar tracks */
    TankCaterpillar: 4,
    
    /** Energy shield effect */
    Shield: 6,
    
    /** Tank turret on top of hull */
    TankTurret: 8,
    
    /** Bullets flying through the air */
    Bullet: 1,
    
    // === Effects without shadow mapping ===
    
    /** Explosion effects */
    Explosion: 100,
    
    /** Hit flash effects */
    HitFlash: 100,
    
    /** Muzzle flash from firing */
    MuzzleFlash: 100,
} as const;

export type ZIndexType = typeof ZIndexConfig;
export type ZIndexKey = keyof ZIndexType;

