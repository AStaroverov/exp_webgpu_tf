/**
 * Physics Configuration
 * 
 * Collision groups, physics damping, and physical world constants.
 */

// =============================================================================
// COLLISION GROUPS
// =============================================================================

/**
 * Collision groups determine which entities can collide with each other.
 * Uses bitmasking for efficient collision detection.
 */
export const CollisionGroupConfig = {
    /** No collision */
    NONE: 0,
    
    /** Collides with everything */
    ALL: 0xFFFF,
    
    /** Static obstacles (walls, rocks) */
    OBSTACLE: 0b00000001,
    
    /** Projectiles (bullets, missiles) */
    BULLET: 0b00000010,
    
    /** Vehicle base (main body without parts) */
    VEHICLE_BASE: 0b00000100,
    
    /** Vehicle hull parts (destructible armor) */
    VEHICLE_HULL_PARTS: 0b00001000,
    
    /** Tank turret head parts */
    TANK_TURRET_HEAD_PARTS: 0b00100000,
    
    /** Tank turret gun barrel parts */
    TANK_TURRET_GUN_PARTS: 0b01000000,
    
    /** Energy shield (blocks only bullets) */
    SHIELD: 0b10000000,
    
    /** Spice resources (interacts only with harvester scoop and collectors) */
    SPICE: 0b100000000,
    
    /** Spice collector sensor (detects spice intersection) */
    SPICE_COLLECTOR: 0b1000000000,
} as const;

export const ALL_VEHICLE_PARTS_MASK = 
    CollisionGroupConfig.VEHICLE_HULL_PARTS | 
    CollisionGroupConfig.TANK_TURRET_HEAD_PARTS | 
    CollisionGroupConfig.TANK_TURRET_GUN_PARTS;

export const TANK_PARTS_MASK = 
    CollisionGroupConfig.VEHICLE_HULL_PARTS | 
    CollisionGroupConfig.TANK_TURRET_HEAD_PARTS | 
    CollisionGroupConfig.TANK_TURRET_GUN_PARTS;

// =============================================================================
// PHYSICS DAMPING
// =============================================================================

/**
 * Default damping values for different entity types.
 * Higher values = more friction/slower movement.
 */
export const DampingConfig: {
    vehicleLinear: number;
    vehicleAngular: number;
    carLinear: number;
    carAngular: number;
    bulletLinear: number;
    bulletAngular: number;
} = {
    /** Default linear damping for vehicles */
    vehicleLinear: 5,
    
    /** Default angular damping for vehicles */
    vehicleAngular: 5,
    
    /** Linear damping for fast cars (less friction) */
    carLinear: 3,
    
    /** Angular damping for cars (quick turning) */
    carAngular: 4,
    
    /** Linear damping for bullets (almost none) */
    bulletLinear: 0.1,
    
    /** Angular damping for bullets */
    bulletAngular: 0.1,
};

// =============================================================================
// MOVEMENT PHYSICS
// =============================================================================

/**
 * Movement impulse factors for vehicle control.
 * These multiply with delta time to create smooth movement.
 */
export const MovementConfig = {
    /** Base impulse factor for forward/backward movement */
    baseImpulseFactor: 15_000_000_000,
    
    /** Base torque factor for rotation */
    baseRotationImpulseFactor: 150_000_000_000,
} as const;

export type CollisionGroupType = typeof CollisionGroupConfig;
export type DampingType = typeof DampingConfig;
export type MovementType = typeof MovementConfig;

