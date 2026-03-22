import { cos, sin, log1p, abs, sign } from '../../lib/math.ts';
import { random, randomRangeInt } from '../../lib/random.ts';
import { shuffle } from '../../lib/shuffle.ts';
import { throwingError } from '../../lib/throwingError.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    RAY_FEATURES_DIM,
    RAY_SLOTS,
    TANK_FEATURES_DIM,
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

import { HISTORY_LENGTH, HISTORY_OFFSETS } from './historyConfig.ts';
export { HISTORY_LENGTH, HISTORY_OFFSETS };
export type StateHistory = InputArrays[];  // always length HISTORY_LENGTH: [t, t-3, t-6, t-9, t-12]

export type InputArrays = {
    battleFeatures: Float32Array,
    
    tankType: Int32Array,          // tank type for embedding
    tankFeatures: Float32Array,
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
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
): InputArrays {
    // ---- Battle features ----
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM);
    let bi = 0;

    battleFeatures[bi++] = log1p(width);
    battleFeatures[bi++] = log1p(height);

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

        turretFeatures[dstOffset + 0] = norm(turretX - tankX, QUANT);
        turretFeatures[dstOffset + 1] = norm(turretY - tankY, QUANT);
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
        enemiesFeatures[dstOffset + 1] = norm(eX - tankX, QUANT);
        enemiesFeatures[dstOffset + 2] = norm(eY - tankY, QUANT);
        enemiesFeatures[dstOffset + 3] = norm(eVx - speedX, QUANT);
        enemiesFeatures[dstOffset + 4] = norm(eVy - speedY, QUANT);
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
        alliesFeatures[dstOffset + 1] = norm(aX - tankX, QUANT);
        alliesFeatures[dstOffset + 2] = norm(aY - tankY, QUANT);
        alliesFeatures[dstOffset + 3] = norm(aVx - speedX, QUANT);
        alliesFeatures[dstOffset + 4] = norm(aVy - speedY, QUANT);
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
            tankX, tankY,
        );
    }

    const result = {
        battleFeatures,
        
        tankType,
        tankFeatures,
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
    };

    if (!checkInputArrays(result)) {
        throw new Error('Invalid input arrays');
    }

    return result;
}

export function assembleStateHistory(
    episodeStates: InputArrays[],
    stepIndex: number,
): StateHistory {
    const history: InputArrays[] = [];
    for (let f = 0; f < HISTORY_LENGTH; f++) {
        const targetIdx = stepIndex - HISTORY_OFFSETS[f];
        const clampedIdx = Math.max(0, targetIdx);
        history.push(episodeStates[clampedIdx]);
    }
    return history;
}

/**
 * Собирает StateHistory для текущего шага без копирования всего массива.
 * Эквивалент assembleStateHistory([...pastStates, currentState], pastStates.length),
 * но O(HISTORY_LENGTH) вместо O(N).
 */
export function assembleCurrentStateHistory(
    pastStates: InputArrays[],
    currentState: InputArrays,
): StateHistory {
    const lastIdx = pastStates.length;
    const history: InputArrays[] = [];
    for (let f = 0; f < HISTORY_LENGTH; f++) {
        const targetIdx = Math.max(0, lastIdx - HISTORY_OFFSETS[f]);
        history.push(targetIdx >= pastStates.length ? currentState : pastStates[targetIdx]);
    }
    return history;
}

export function prepareRandomStateHistory(): StateHistory {
    return Array.from({ length: HISTORY_LENGTH }, () => prepareRandomInputArrays());
}

export function prepareRandomInputArrays(): InputArrays {
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM).map(() => random());
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM).map(() => random());
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
    
    return {
        battleFeatures,

        tankFeatures,
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
    tankX: number,
    tankY: number,
): void {
    const hitType = raysBuffer[srcOffset + 0];
    const rootX = raysBuffer[srcOffset + 2];
    const rootY = raysBuffer[srcOffset + 3];
    const dirX = raysBuffer[srcOffset + 4];
    const dirY = raysBuffer[srcOffset + 5];
    const distance = raysBuffer[srcOffset + 6];

    raysFeatures[dstOffset + 0] = norm(rootX - tankX, QUANT);
    raysFeatures[dstOffset + 1] = norm(rootY - tankY, QUANT);
    raysFeatures[dstOffset + 2] = dirX;
    raysFeatures[dstOffset + 3] = dirY;
    raysFeatures[dstOffset + 4] = norm(distance, QUANT);
    // hit obstacle encoding: 0 = none, -1 = obstacle, 1 = vehicle
    raysFeatures[dstOffset + 5] = hitType === RayHitType.NONE
        ? 0
        : hitType === RayHitType.OBSTACLE
            ? -1
            : hitType === RayHitType.ENEMY_VEHICLE || hitType === RayHitType.ALLY_VEHICLE
                ? 1
                : throwingError(`Invalid hit type: ${hitType}`);
    // ally/enemy encoding: 1 = ally, -1 = enemy, 0 = other
    raysFeatures[dstOffset + 6] = hitType === RayHitType.ALLY_VEHICLE
        ? 1
        : hitType === RayHitType.ENEMY_VEHICLE
            ? -1
            : 0;
}
