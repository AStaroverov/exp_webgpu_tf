import { cos, sin, log1p, abs, sign } from '../../../../lib/math.ts';
import { random, randomRangeInt } from '../../../../lib/random.ts';
import { shuffle } from '../../../../lib/shuffle.ts';
import { throwingError } from '../../../../lib/throwingError.ts';
import { AgentMemory } from '../../../ppo/src/memory/Memory.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    GRID_CELLS,
    GRID_CELL_FEATURES,
    GRID_SIZE,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
    TANK_HISTORY_STEPS,
    TANK_HISTORY_FEATURE_DIM,
    TURRET_FEATURES_DIM,
} from '../models/dims.ts';
import {
    TankInputTensor,
    RayHitType,
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    RAY_BUFFER,
    MAX_TURRETS,
    TURRET_BUFFER,
    RAYS_COUNT
} from '../../../tanks/src/Plugins/Pilots/Components/TankState.ts';

function norm(v: number, size: number): number {
    return v / size;
}

function logNorm(v: number, size: number): number {
    return sign(v) * log1p(abs(v / size));
}

const QUANT = 100;

const ENEMIES_INDEXES = new Uint32Array(Array.from({ length: ENEMY_SLOTS }, (_, i) => i));
const ALLIES_INDEXES = new Uint32Array(Array.from({ length: ALLY_SLOTS }, (_, i) => i));
const BULLETS_INDEXES = new Uint32Array(Array.from({ length: BULLET_SLOTS }, (_, i) => i));

export type InputArrays = {
    tankType: Int32Array,          // tank type for embedding
    tankFeatures: Float32Array,
    tankFeaturesHistory: Float32Array, // [TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM]
    turretFeatures: Float32Array,

    raysFeatures: Float32Array,    // unified env + turret rays
    
    enemiesFeatures: Float32Array,
    enemiesTypes: Int32Array,      // enemy tank types for embedding
    enemiesMask: Float32Array,
    
    alliesFeatures: Float32Array,
    alliesTypes: Int32Array,       // ally tank types for embedding
    alliesMask: Float32Array,
    
    bulletsFeatures: Float32Array,
    bulletsMask: Float32Array,

    obstacleGrid: Float32Array,  // GRID_CELLS * GRID_CELL_FEATURES
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
    obstacleGrid: Float32Array,
    memory: AgentMemory<InputArrays> | null,
): InputArrays {
    // ---- Tank features ----
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM);
    let ti = 0;

    const tankX = TankInputTensor.position.get(tankEid, 0);
    const tankY = TankInputTensor.position.get(tankEid, 1);
    const rotation = TankInputTensor.rotation[tankEid];
    const speedX = TankInputTensor.speed.get(tankEid, 0);
    const speedY = TankInputTensor.speed.get(tankEid, 1);
    const colliderRadius = TankInputTensor.colliderRadius[tankEid];

    tankFeatures[ti++] = TankInputTensor.health[tankEid]; // hp
    tankFeatures[ti++] = norm(tankX, width / 2); // x
    tankFeatures[ti++] = norm(tankY, height / 2); // y
    tankFeatures[ti++] = cos(rotation); // body rotation cos
    tankFeatures[ti++] = sin(rotation); // body rotation sin
    tankFeatures[ti++] = norm(speedX, QUANT); // norm(locSpeedX, QUANT); // speeX
    tankFeatures[ti++] = norm(speedY, QUANT); // norm(locSpeedY, QUANT); // speeY
    tankFeatures[ti++] = logNorm(colliderRadius, QUANT); // collider radis

    const tankType = new Int32Array(1);
    tankType[0] = TankInputTensor.tankType[tankEid];
    
    // ---- Turrets features ----
    const turretFeatures = new Float32Array(MAX_TURRETS * TURRET_FEATURES_DIM);
    const turretsData = TankInputTensor.turretsData.getBatch(tankEid);

    for (let i = 0; i < MAX_TURRETS; i++) {
        const dstOffset = i * TURRET_FEATURES_DIM;
        const srcOffset = i * TURRET_BUFFER;

        if (turretsData[srcOffset + 0] === 0) {
            continue;
        }

        const turretRotation = turretsData[srcOffset + 3];

        turretFeatures[dstOffset + 0] = cos(turretRotation - rotation);
        turretFeatures[dstOffset + 1] = sin(turretRotation - rotation);
    }
    
    // ---- Enemies features ----
    const enemiesMask = new Float32Array(ENEMY_SLOTS);
    const enemiesTypes = new Int32Array(ENEMY_SLOTS);
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM);
    const enemiesBuffer = TankInputTensor.enemiesData.getBatch(tankEid);

    shuffle(ENEMIES_INDEXES);

    for (let r = 0; r < ENEMY_SLOTS; r++) {
        const w = ENEMIES_INDEXES[r];
        const srcOffset = r * ENEMY_BUFFER;

        if (enemiesBuffer[srcOffset] === 0) {
            continue;
        }

        enemiesMask[w] = 1;
        enemiesTypes[w] = enemiesBuffer[srcOffset + 1];
        encodeUnitFeatures(
            enemiesBuffer, srcOffset,
            enemiesFeatures, w * ENEMY_FEATURES_DIM,
            tankX, tankY,
            speedX, speedY,
        );
    }

    // ---- Allies features ----
    const alliesMask = new Float32Array(ALLY_SLOTS);
    const alliesTypes = new Int32Array(ALLY_SLOTS);
    const alliesFeatures = new Float32Array(ALLY_SLOTS * ALLY_FEATURES_DIM);
    const alliesBuffer = TankInputTensor.alliesData.getBatch(tankEid);

    shuffle(ALLIES_INDEXES);

    for (let r = 0; r < ALLY_SLOTS; r++) {
        const w = ALLIES_INDEXES[r];
        const srcOffset = r * ALLY_BUFFER;

        if (alliesBuffer[srcOffset] === 0) {
            continue;
        }

        alliesMask[w] = 1;
        alliesTypes[w] = alliesBuffer[srcOffset + 1];
        encodeUnitFeatures(
            alliesBuffer, srcOffset,
            alliesFeatures, w * ALLY_FEATURES_DIM,
            tankX, tankY, speedX, speedY,
        );
    }

    // ---- Bullets features ----
    const bulletsMask = new Float32Array(BULLET_SLOTS);
    const bulletsFeatures = new Float32Array(BULLET_SLOTS * BULLET_FEATURES_DIM);
    const bulletsBuffer = TankInputTensor.bulletsData.getBatch(tankEid);

    shuffle(BULLETS_INDEXES);

    for (let r = 0; r < BULLET_SLOTS; r++) {
        const w = BULLETS_INDEXES[r];
        const dstOffset = w * BULLET_FEATURES_DIM;
        const srcOffset = r * BULLET_BUFFER;

        if (bulletsBuffer[srcOffset] === 0) {
            continue;
        }

        const bX = bulletsBuffer[srcOffset + 1];
        const bY = bulletsBuffer[srcOffset + 2];
        const bSpeedX = bulletsBuffer[srcOffset + 3];
        const bSpeedY = bulletsBuffer[srcOffset + 4];

        bulletsMask[w] = 1;
        // Relative to tank
        bulletsFeatures[dstOffset + 0] = norm(bX - tankX, QUANT);
        bulletsFeatures[dstOffset + 1] = norm(bY - tankY, QUANT);
        bulletsFeatures[dstOffset + 2] = norm(bSpeedX - speedX, QUANT);
        bulletsFeatures[dstOffset + 3] = norm(bSpeedY - speedY, QUANT);
    }

    // ---- Unified rays features (environment + turret rays) ----
    const raysFeatures = new Float32Array(RAY_SLOTS * RAY_FEATURES_DIM);
    const raysBuffer = TankInputTensor.raysData.getBatch(tankEid);

    // Process environment rays
    for (let r = 0; r < RAYS_COUNT; r++) {
        const dstOffset = r * RAY_FEATURES_DIM;
        const srcOffset = r * RAY_BUFFER;
        encodeRayFeatures(
            raysBuffer, srcOffset,
            raysFeatures, dstOffset,
        );
    }

    // ---- Tank history (pure, derived from memory.states) ----
    const tankFeaturesHistory = buildTankHistory(memory, tankX, tankY, width, height);

    const result = {
        tankType,
        tankFeatures,
        tankFeaturesHistory,
        turretFeatures,

        raysFeatures,
        
        enemiesFeatures,
        enemiesTypes,
        enemiesMask,
        
        alliesFeatures,
        alliesTypes,
        alliesMask,
        
        bulletsFeatures,
        bulletsMask,

        obstacleGrid: encodeObstacleGrid(obstacleGrid, tankX, tankY, width, height),
    };

    if (!checkInputArrays(result)) {
        throw new Error('Invalid input arrays');
    }

    return result;
}

export function prepareRandomInputArrays(): InputArrays {
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM).map(() => random());
    const tankFeaturesHistory = new Float32Array(TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM).map(() => random());
    const tankType = new Int32Array(1).map(() => randomRangeInt(0, 5));
    const raysFeatures = new Float32Array(RAY_SLOTS * RAY_FEATURES_DIM).map(() => random());
    
    const turretFeatures = new Float32Array(MAX_TURRETS * TURRET_FEATURES_DIM).map(() => random());

    const enemiesMask = new Float32Array(ENEMY_SLOTS).map(() => randomRangeInt(0, 1));
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM).map(() => random());
    const enemiesTypes = new Int32Array(ENEMY_SLOTS).map(() => randomRangeInt(0, 5));

    const alliesMask = new Float32Array(ALLY_SLOTS).map(() => randomRangeInt(0, 1));
    const alliesFeatures = new Float32Array(ALLY_SLOTS * ALLY_FEATURES_DIM).map(() => random());
    const alliesTypes = new Int32Array(ALLY_SLOTS).map(() => randomRangeInt(0, 5));

    const bulletsMask = new Float32Array(BULLET_SLOTS).map(() => randomRangeInt(0, 1));
    const bulletsFeatures = new Float32Array(BULLET_SLOTS * BULLET_FEATURES_DIM).map(() => random());

    const obstacleGrid = new Float32Array(GRID_CELLS * GRID_CELL_FEATURES).map(() => random());

    return {
        tankFeatures,
        tankFeaturesHistory,
        tankType,
        turretFeatures,

        raysFeatures,

        enemiesFeatures,
        enemiesTypes,
        enemiesMask,

        alliesFeatures,
        alliesTypes,
        alliesMask,

        bulletsFeatures,
        bulletsMask,

        obstacleGrid,
    };
}

export function checkInputArrays(inputArray: InputArrays): boolean {
    return Object.values(inputArray).every(arr => arr.every(v => !isNaN(v)));
}

function encodeRayFeatures(
    raysBuffer: Float64Array,
    srcOffset: number,
    raysFeatures: Float32Array,
    dstOffset: number,
): void {
    const hitType = raysBuffer[srcOffset + 0];
    const dirX = raysBuffer[srcOffset + 4];
    const dirY = raysBuffer[srcOffset + 5];
    const distance = raysBuffer[srcOffset + 6];
    
    // Direction and distance (relative to ray root on tank)
    raysFeatures[dstOffset + 0] = norm(dirX * distance, QUANT);
    raysFeatures[dstOffset + 1] = norm(dirY * distance, QUANT);
    // hit obstacle encoding: 0 = none, -1 = obstacle, 1 = vehicle
    raysFeatures[dstOffset + 2] = hitType === RayHitType.NONE
        ? 0
        : hitType === RayHitType.OBSTACLE
            ? -1
            : hitType === RayHitType.ENEMY_VEHICLE || hitType === RayHitType.ALLY_VEHICLE
                ? 1
                : throwingError(`Invalid hit type: ${hitType}`);
    // ally/enemy encoding: 1 = ally, -1 = enemy, 0 = other
    raysFeatures[dstOffset + 3] = hitType === RayHitType.ALLY_VEHICLE
        ? 1
        : hitType === RayHitType.ENEMY_VEHICLE
            ? -1
            : 0;
}

function encodeUnitFeatures(
    buffer: Float64Array, srcOffset: number,
    features: Float32Array, dstOffset: number,
    tankX: number, tankY: number,
    speedX: number, speedY: number,
): void {
    const hp = buffer[srcOffset + 2];
    const x = buffer[srcOffset + 3];
    const y = buffer[srcOffset + 4];
    const vx = buffer[srcOffset + 5];
    const vy = buffer[srcOffset + 6];
    const colliderRadius = buffer[srcOffset + 8];

    // Relative to tank
    features[dstOffset + 0] = hp;
    features[dstOffset + 1] = norm(x - tankX, QUANT);
    features[dstOffset + 2] = norm(y - tankY, QUANT);
    features[dstOffset + 3] = norm(vx - speedX, QUANT);
    features[dstOffset + 4] = norm(vy - speedY, QUANT);
    // Collider radius
    features[dstOffset + 5] = logNorm(colliderRadius, QUANT);
}

function encodeObstacleGrid(
    rawGrid: Float32Array,
    tankX: number, tankY: number,
    width: number, height: number,
): Float32Array {
    const buf = new Float32Array(GRID_CELLS * GRID_CELL_FEATURES);
    const cellW = width / GRID_SIZE;
    const cellH = height / GRID_SIZE;
    let offset = 0;
    for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
            const cellX = (col + 0.5) * cellW;
            const cellY = (row + 0.5) * cellH;
            buf[offset++] = rawGrid[row * GRID_SIZE + col];
            buf[offset++] = norm(cellX - tankX, width / 2);
            buf[offset++] = norm(cellY - tankY, height / 2);
        }
    }
    return buf;
}

const HISTORY_STRIDE = 5;

function buildTankHistory(
    memory: AgentMemory<InputArrays> | null,
    tankX: number, tankY: number,
    width: number, height: number,
): Float32Array {
    const out = new Float32Array(TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM);
    if (!memory || memory.states.length === 0) return out;

    const states = memory.states;
    let filled = 0;

    for (let i = 0; i < TANK_HISTORY_STEPS; i++) {
        const idx = states.length - 1 - i * HISTORY_STRIDE;
        if (idx < 0) break;
        filled++;
        const prev = states[idx].tankFeatures;
        const hx = prev[1] * (width / 2);  // denorm
        const hy = prev[2] * (height / 2);
        const off = i * TANK_HISTORY_FEATURE_DIM;
        out[off + 0] = norm(tankX - hx, QUANT);
        out[off + 1] = norm(tankY - hy, QUANT);
    }

    // replicate oldest known into remaining slots
    if (filled > 0 && filled < TANK_HISTORY_STEPS) {
        const srcOff = (filled - 1) * TANK_HISTORY_FEATURE_DIM;
        for (let i = filled; i < TANK_HISTORY_STEPS; i++) {
            out.set(out.subarray(srcOff, srcOff + TANK_HISTORY_FEATURE_DIM), i * TANK_HISTORY_FEATURE_DIM);
        }
    }

    return out;
}