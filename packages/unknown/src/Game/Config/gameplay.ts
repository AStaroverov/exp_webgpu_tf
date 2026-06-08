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
    decayDistanceThreshold: 0,
    
    /** Probability per frame for decay check */
    decayProbability: 0.3,
    
    /** Probability per part to be destroyed during decay */
    partDecayProbability: 0.05,
} as const;

// =============================================================================
// SPOTTING
// =============================================================================

export const SpottingConfig = {
    /** Proximity reveal around every unit, hex steps */
    spotRadius: 2,

    /** Ranger searchlight reach, hex cells */
    beamLength: 4,

    /** Reveal on firing when dist < this (<= 4) */
    fireRevealDist: 3,

    /** Confidence 1 -> 0 after losing the spot (ms) */
    memoryMs: 5000,

    /**
     * Confidence a spot is reinforced TO, graded by source (the value decays from
     * here over `memoryMs`). A unit refreshes to the MAX active source each tick,
     * so a beam-lit unit reads 1 even if it is also merely near another. Any
     * confidence > 0 counts as "visible" (discrete Enemy plane); the value itself
     * scales heat / the SpotConfidence channel / the spotting reward.
     *   beam      — Ranger searchlight: a full firing-quality lock.
     *   fire      — muzzle-flash self-reveal: a medium, fading giveaway.
     *   proximity — own-sensor blip within `spotRadius`: a weak, vague contact.
     */
    confidence: {
        beam: 1,
        fire: 0.5,
        proximity: 0.25,
    },
} as const;

export type ShieldType = typeof ShieldConfig;
export type GameZoneType = typeof GameZoneConfig;
export type SpottingType = typeof SpottingConfig;

