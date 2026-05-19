import {
    MAX_ALLIES,
    MAX_BULLETS,
    MAX_ENEMIES,
    RAYS_COUNT,
    RAY_HIT_TYPE_COUNT,
    MAX_TURRETS,
} from '../../../tanks/src/Plugins/Pilots/Components/TankState.ts';

export { RAY_HIT_TYPE_COUNT }; // LightTank, MediumTank, HeavyTank, PlayerTank, Harvester, MeleeCar

export const TANK_FEATURES_DIM = 8;

// Tank history: [relX, relY] per step
export const TANK_HISTORY_STEPS = 6;
export const TANK_HISTORY_FEATURE_DIM = 2;

export const TURRET_SLOTS = MAX_TURRETS;
export const TURRET_FEATURES_DIM = 2;

// Enemies: [hp, relX, relY, relVx, relVy, colliderRadius]
export const ENEMY_SLOTS = MAX_ENEMIES;
export const ENEMY_FEATURES_DIM = 6;

// Allies: [hp, relX, relY, relVx, relVy, colliderRadius]
export const ALLY_SLOTS = MAX_ALLIES;
export const ALLY_FEATURES_DIM = 6;

// Bullets: [relX, relY, relVx, relVy]
export const BULLET_SLOTS = MAX_BULLETS;
export const BULLET_FEATURES_DIM = 4;

// Unified rays (environment + turret rays combined)
export const RAY_SLOTS = RAYS_COUNT;
export const RAY_FEATURES_DIM = 4; // [distanceX, distanceY, hitObstacle, hitVehicle]

// Obstacle spatial map (16×16 grid)
export const GRID_SIZE = 16;
export const GRID_CELL_FEATURES = 3;  // obstacle + rel_x + rel_y
export const GRID_CELLS = GRID_SIZE * GRID_SIZE;  // 256

export const ACTION_HEAD_DIMS = [15, 15, 2, 31];
