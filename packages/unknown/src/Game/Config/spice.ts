/**
 * Spice Configuration
 * 
 * Configuration for spice resources that can be collected by harvesters.
 */

import { TColor } from '../../../../renderer/src/ECS/Components/Common.ts';
import { ZIndexConfig } from './zindex.ts';
import { ALL_VEHICLE_PARTS_MASK, CollisionGroupConfig } from './physics.ts';

// =============================================================================
// SPICE PROPERTIES
// =============================================================================

export const SpiceConfig = {
    /** Size of each spice piece */
    size: [4, 14],
    
    /** Density will be calculated based on the size */
    density: 20,
    
    /** Default damping for spice movement */
    damping: 8,
    
    /** Default count range for cluster spawn [min, max] */
    countRange: [50, 150] as [number, number],
    
    /** Default spread range for cluster spawn [min, max] */
    spreadRange: [20, 100] as [number, number],
} as const;

// =============================================================================
// SPICE PHYSICS CONFIGURATION
// =============================================================================

export const SpicePhysicsConfig = {
    /** Z-index for rendering order */
    z: ZIndexConfig.Spice,
    /** Solver group membership */
    belongsSolverGroup: CollisionGroupConfig.ALL,
    /** Solver group interactions */
    interactsSolverGroup: CollisionGroupConfig.ALL,
    /** Collision group membership */
    belongsCollisionGroup: CollisionGroupConfig.SPICE,
    /** Collision group interactions (spice + harvester scoop + collectors) */
    interactsCollisionGroup: 
        CollisionGroupConfig.SPICE | 
        CollisionGroupConfig.VEHICLE_HULL_PARTS |
        CollisionGroupConfig.SPICE_COLLECTOR,
} as const;

// =============================================================================
// SPICE COLLECTOR PHYSICS CONFIGURATION
// =============================================================================

export const SpiceCollectorPhysicsConfig = {
    /** Z-index for rendering order */
    z: ZIndexConfig.Spice,
    /** Collision group membership */
    belongsCollisionGroup: CollisionGroupConfig.SPICE_COLLECTOR,
    /** Collision group interactions (spice + vehicle parts/debris) */
    interactsCollisionGroup: CollisionGroupConfig.SPICE | ALL_VEHICLE_PARTS_MASK,
    /** Solver groups - no physics response needed */
    belongsSolverGroup: CollisionGroupConfig.NONE,
    interactsSolverGroup: CollisionGroupConfig.NONE,
} as const;

// =============================================================================
// SPICE COLORS
// =============================================================================

/** Spice colors - various warm orange/brown shades (like Dune's melange) */
export const SpiceColors: TColor[] = [
    new Float32Array([0.9, 0.5, 0.1, 1]),    // Orange
    new Float32Array([0.85, 0.45, 0.15, 1]), // Dark orange
    new Float32Array([0.95, 0.55, 0.2, 1]),  // Light orange
    new Float32Array([0.8, 0.4, 0.1, 1]),    // Brown-orange
    new Float32Array([0.92, 0.6, 0.25, 1]),  // Golden orange
    new Float32Array([0.88, 0.48, 0.12, 1]), // Amber
];

export type SpiceType = typeof SpiceConfig;

