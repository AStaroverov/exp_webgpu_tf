/**
 * Gameplay Configuration
 * 
 * Game rules, timing, regeneration, and other gameplay-related settings.
 */

// =============================================================================
// SHIELD REGENERATION
// =============================================================================

export const ShieldConfig = {
    /** Time between shield element regeneration (ms) */
    regenInterval: 100,
} as const;

// =============================================================================
// GAME ZONE
// =============================================================================

export const GameZoneConfig = {
    /** Padding outside the game zone before entities are destroyed */
    destructionPadding: 500,
    
    /** Distance beyond game zone where vehicle decay starts */
    decayDistanceThreshold: 400,
    
    /** Probability per frame for decay check */
    decayProbability: 0.3,
    
    /** Probability per part to be destroyed during decay */
    partDecayProbability: 0.05,
} as const;

export type ShieldType = typeof ShieldConfig;
export type GameZoneType = typeof GameZoneConfig;

