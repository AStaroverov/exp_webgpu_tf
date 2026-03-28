import { cos, sin, log1p, abs, sign } from '../../lib/math.ts';
import { random, randomRangeInt } from '../../lib/random.ts';
import { shuffle } from '../../lib/shuffle.ts';
import { throwingError } from '../../lib/throwingError.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    GRID_CELLS,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
    TANK_HISTORY_STEPS,
    TANK_HISTORY_FEATURE_DIM,
    TURRET_FEATURES_DIM,
} from '../ml/src/Models/Create.ts';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    RAY_BUFFER, MAX_TURRETS,
    RayHitType,
    TURRET_BUFFER, TankInputTensor,
    RAYS_COUNT
} from '../tanks/src/Plugins/Pilots/Components/TankState.ts';

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

    obstacleGrid: Float32Array,  // GRID_CELLS = 256
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
    obstacleGrid: Float32Array,
    historyBuffer: TankHistoryBuffer,
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

        const turretX = turretsData[srcOffset + 1];
        const turretY = turretsData[srcOffset + 2];
        const turretRotation = turretsData[srcOffset + 3];

        turretFeatures[dstOffset + 0] = norm(turretX, width / 2);
        turretFeatures[dstOffset + 1] = norm(turretY, height / 2);
        turretFeatures[dstOffset + 2] = cos(turretRotation);
        turretFeatures[dstOffset + 3] = sin(turretRotation);
    }
    
    // ---- Enemies features ----
    const enemiesMask = new Float32Array(ENEMY_SLOTS);
    const enemiesTypes = new Int32Array(ENEMY_SLOTS);
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM);
    const enemiesBuffer = TankInputTensor.enemiesData.getBatch(tankEid);

    shuffle(ENEMIES_INDEXES);

    for (let r = 0; r < ENEMY_SLOTS; r++) {
        const w = ENEMIES_INDEXES[r];
        const dstOffset = w * ENEMY_FEATURES_DIM;
        const srcOffset = r * ENEMY_BUFFER;

        if (enemiesBuffer[srcOffset] === 0) {
            continue;
        }

        const eType = enemiesBuffer[srcOffset + 1];
        const hp = enemiesBuffer[srcOffset + 2];
        const eX = enemiesBuffer[srcOffset + 3];
        const eY = enemiesBuffer[srcOffset + 4];
        const eVx = enemiesBuffer[srcOffset + 5];
        const eVy = enemiesBuffer[srcOffset + 6];
        const eTurretRotation = enemiesBuffer[srcOffset + 7];
        const eColliderRadius = enemiesBuffer[srcOffset + 8];

        enemiesMask[w] = 1;
        enemiesTypes[w] = eType;
        enemiesFeatures[dstOffset + 0] = hp;
        enemiesFeatures[dstOffset + 1] = norm(eX, width / 2);
        enemiesFeatures[dstOffset + 2] = norm(eY, height / 2);
        enemiesFeatures[dstOffset + 3] = norm(eVx, QUANT);
        enemiesFeatures[dstOffset + 4] = norm(eVy, QUANT);
        enemiesFeatures[dstOffset + 5] = cos(eTurretRotation);
        enemiesFeatures[dstOffset + 6] = sin(eTurretRotation);
        enemiesFeatures[dstOffset + 7] = logNorm(eColliderRadius, QUANT);
    }

    // ---- Allies features ----
    const alliesMask = new Float32Array(ALLY_SLOTS);
    const alliesTypes = new Int32Array(ALLY_SLOTS);
    const alliesFeatures = new Float32Array(ALLY_SLOTS * ALLY_FEATURES_DIM);
    const alliesBuffer = TankInputTensor.alliesData.getBatch(tankEid);

    shuffle(ALLIES_INDEXES);

    for (let r = 0; r < ALLY_SLOTS; r++) {
        const w = ALLIES_INDEXES[r];
        const dstOffset = w * ALLY_FEATURES_DIM;
        const srcOffset = r * ALLY_BUFFER;

        if (alliesBuffer[srcOffset] === 0) {
            continue;
        }

        const aType = alliesBuffer[srcOffset + 1];
        const hp = alliesBuffer[srcOffset + 2];
        const aX = alliesBuffer[srcOffset + 3];
        const aY = alliesBuffer[srcOffset + 4];
        const aVx = alliesBuffer[srcOffset + 5];
        const aVy = alliesBuffer[srcOffset + 6];
        const aTurretRotation = alliesBuffer[srcOffset + 7];
        const aColliderRadius = alliesBuffer[srcOffset + 8];

        alliesMask[w] = 1;
        alliesTypes[w] = aType;
        alliesFeatures[dstOffset + 0] = hp;
        alliesFeatures[dstOffset + 1] = norm(aX, width / 2);
        alliesFeatures[dstOffset + 2] = norm(aY, height / 2);
        alliesFeatures[dstOffset + 3] = norm(aVx, QUANT);
        alliesFeatures[dstOffset + 4] = norm(aVy, QUANT);
        alliesFeatures[dstOffset + 5] = cos(aTurretRotation);
        alliesFeatures[dstOffset + 6] = sin(aTurretRotation);
        alliesFeatures[dstOffset + 7] = logNorm(aColliderRadius, QUANT);
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

    // ---- Tank history ----
    const historyValues = new Float32Array(TANK_HISTORY_FEATURE_DIM);
    historyValues[0] = norm(tankX, width / 2);
    historyValues[1] = norm(tankY, height / 2);

    const tankFeaturesHistory = historyBuffer
        ? historyBuffer.update(tankEid, historyValues)
        : new Float32Array(TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM);

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

        obstacleGrid,
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

    const obstacleGrid = new Float32Array(GRID_CELLS).map(() => randomRangeInt(0, 1));

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

    raysFeatures[dstOffset + 0] = dirX;
    raysFeatures[dstOffset + 1] = dirY;
    raysFeatures[dstOffset + 2] = norm(distance, QUANT);
    // hit obstacle encoding: 0 = none, -1 = obstacle, 1 = vehicle
    raysFeatures[dstOffset + 3] = hitType === RayHitType.NONE
        ? 0
        : hitType === RayHitType.OBSTACLE
            ? -1
            : hitType === RayHitType.ENEMY_VEHICLE || hitType === RayHitType.ALLY_VEHICLE
                ? 1
                : throwingError(`Invalid hit type: ${hitType}`);
    // ally/enemy encoding: 1 = ally, -1 = enemy, 0 = other
    raysFeatures[dstOffset + 4] = hitType === RayHitType.ALLY_VEHICLE
        ? 1
        : hitType === RayHitType.ENEMY_VEHICLE
            ? -1
            : 0;
}

const HISTORY_STRIDE = 5; // sample every 5th frame

export class TankHistoryBuffer {
    private buffers = new Map<number, Float32Array>();
    private counters = new Map<number, number>();
    private filled = new Map<number, number>();

    update(tankEid: number, values: Float32Array): Float32Array {
        const size = TANK_HISTORY_STEPS * TANK_HISTORY_FEATURE_DIM;
        let buf = this.buffers.get(tankEid);
        if (!buf) {
            buf = new Float32Array(size);
            this.buffers.set(tankEid, buf);
            this.filled.set(tankEid, 0);
        }

        const count = (this.counters.get(tankEid) ?? 0) + 1;
        this.counters.set(tankEid, count);

        if (count % HISTORY_STRIDE === 0) {
            // shift right: [s0, s1, s2, ...] → [_, s0, s1, ...]
            buf.copyWithin(TANK_HISTORY_FEATURE_DIM, 0, size - TANK_HISTORY_FEATURE_DIM);
            // newest at slot 0
            buf.set(values, 0);

            const filledCount = Math.min((this.filled.get(tankEid) ?? 0) + 1, TANK_HISTORY_STEPS);
            this.filled.set(tankEid, filledCount);

            // replicate oldest known into remaining tail slots
            // e.g. filled=3: [t-5, t-10, t-15, t-15, t-15, t-15]
            if (filledCount < TANK_HISTORY_STEPS) {
                const lastFilledOffset = (filledCount - 1) * TANK_HISTORY_FEATURE_DIM;
                for (let i = filledCount; i < TANK_HISTORY_STEPS; i++) {
                    buf.set(buf.subarray(lastFilledOffset, lastFilledOffset + TANK_HISTORY_FEATURE_DIM), i * TANK_HISTORY_FEATURE_DIM);
                }
            }
        }

        return new Float32Array(buf);
    }

    clear() {
        this.buffers.clear();
        this.counters.clear();
        this.filled.clear();
    }
}