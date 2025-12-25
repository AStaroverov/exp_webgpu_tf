import { cos, round, sin } from '../../lib/math.ts';
import { random, randomRangeInt } from '../../lib/random.ts';
import { shuffle } from '../../lib/shuffle.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    ENV_RAY_FEATURES_DIM,
    ENV_RAY_SLOTS,
    TANK_FEATURES_DIM,
    TURRET_RAY_FEATURES_DIM,
    TURRET_RAY_SLOTS,
} from '../ml/src/Models/Create.ts';
import {
    ALLY_BUFFER,
    BULLET_BUFFER,
    ENEMY_BUFFER,
    ENV_RAY_BUFFER, TURRET_RAY_BUFFER, TankInputTensor
} from '../tanks/src/Pilots/Components/TankState.ts';

function norm(v: number, size: number): number {
    return v / size;
}

function rotateVector(x: number, y: number, angle: number): [number, number] {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [x * cos - y * sin, x * sin + y * cos];
}

const QUANT = 100;
const ENEMIES_INDEXES = new Uint32Array(Array.from({ length: ENEMY_SLOTS }, (_, i) => i));
const ALLIES_INDEXES = new Uint32Array(Array.from({ length: ALLY_SLOTS }, (_, i) => i));
const BULLETS_INDEXES = new Uint32Array(Array.from({ length: BULLET_SLOTS }, (_, i) => i));

export type InputArrays = {
    battleFeatures: Float32Array,
    tankFeatures: Float32Array,
    tankType: Int32Array,          // tank type for embedding
    enemiesFeatures: Float32Array,
    enemiesTypes: Int32Array,      // enemy tank types for embedding
    enemiesMask: Float32Array,
    alliesFeatures: Float32Array,
    alliesTypes: Int32Array,       // ally tank types for embedding
    alliesMask: Float32Array,
    bulletsFeatures: Float32Array,
    bulletsMask: Float32Array,
    envRaysFeatures: Float32Array,
    envRaysTypes: Int32Array,      // hit types for embedding (0=none, 1=obstacle, 2=vehicle)
    turretRaysFeatures: Float32Array,
    turretRaysTypes: Int32Array,   // hit types for embedding
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
): InputArrays {
    // ---- Battle features ----
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM);
    let bi = 0;

    battleFeatures[bi++] = Math.log1p(width);
    battleFeatures[bi++] = Math.log1p(height);

    // ---- Tank features ----
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM);
    const tankType = new Int32Array(1);
    tankType[0] = TankInputTensor.tankType[tankEid];
    let ti = 0;

    const tankX = TankInputTensor.position.get(tankEid, 0);
    const tankY = TankInputTensor.position.get(tankEid, 1);
    const rotation = TankInputTensor.rotation[tankEid];
    const speedX = TankInputTensor.speed.get(tankEid, 0);
    const speedY = TankInputTensor.speed.get(tankEid, 1);
    const turretRot = TankInputTensor.turretRotation[tankEid];
    const colliderRadius = TankInputTensor.colliderRadius[tankEid];

    const invRotation = -rotation;
    const [locSpeedX, locSpeedY] = rotateVector(speedX, speedY, invRotation);
    const turretRel = turretRot - rotation;

    tankFeatures[ti++] = TankInputTensor.health[tankEid]; // hp
    tankFeatures[ti++] = norm(tankX, width / 2); // x
    tankFeatures[ti++] = norm(tankY, height / 2); // y
    tankFeatures[ti++] = cos(rotation); // body rotation cos
    tankFeatures[ti++] = sin(rotation); // body rotation sin
    tankFeatures[ti++] = norm(locSpeedX, QUANT); // speedX
    tankFeatures[ti++] = norm(locSpeedY, QUANT); // speedY
    tankFeatures[ti++] = cos(turretRel); // turret rotation cos
    tankFeatures[ti++] = sin(turretRel); // turret rotation sin
    tankFeatures[ti++] = norm(colliderRadius, QUANT); // collider radius

    // ---- Enemies features ----
    // New format: [id, type, hp, x, y] -> features: [hp, locX, locY] + type for embedding
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
        const eX = enemiesBuffer[srcOffset + 3];
        const eY = enemiesBuffer[srcOffset + 4];

        enemiesMask[w] = 1;
        enemiesTypes[w] = eType;
        enemiesFeatures[dstOffset + 0] = enemiesBuffer[srcOffset + 2]; // hp
        enemiesFeatures[dstOffset + 1] = round(norm(eX, width / 2) * 10) / 10;
        enemiesFeatures[dstOffset + 2] = round(norm(eY, width / 2) * 10) / 10;
    }

    // ---- Allies features ----
    // New format: [id, type, hp, x, y] -> features: [hp, locX, locY] + type for embedding
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
        const aX = alliesBuffer[srcOffset + 3];
        const aY = alliesBuffer[srcOffset + 4];

        alliesMask[w] = 1;
        alliesTypes[w] = aType;
        alliesFeatures[dstOffset + 0] = alliesBuffer[srcOffset + 2]; // hp
        alliesFeatures[dstOffset + 1] = round(norm(aX, width / 2) * 10) / 10;
        alliesFeatures[dstOffset + 2] = round(norm(aY, width / 2) * 10) / 10;
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

        const [locX, locY] = rotateVector(bX - tankX, bY - tankY, invRotation);
        const [locSpeedX, locSpeedY] = rotateVector(bSpeedX - speedX, bSpeedY - speedY, invRotation);

        bulletsMask[w] = 1;
        bulletsFeatures[dstOffset + 0] = norm(locX, QUANT);
        bulletsFeatures[dstOffset + 1] = norm(locY, QUANT);
        bulletsFeatures[dstOffset + 2] = norm(locSpeedX, QUANT);
        bulletsFeatures[dstOffset + 3] = norm(locSpeedY, QUANT);
    }

    // ---- Environment rays features ----
    // Format: [rayDirX, rayDirY, hitType, x, y, radius, distance] -> features: [locRayDirX, locRayDirY, locX, locY, radius, distance] + hitType for embedding
    const envRaysFeatures = new Float32Array(ENV_RAY_SLOTS * ENV_RAY_FEATURES_DIM);
    const envRaysTypes = new Int32Array(ENV_RAY_SLOTS);
    const envRaysBuffer = TankInputTensor.envRaysData.getBatch(tankEid);

    for (let r = 0; r < ENV_RAY_SLOTS; r++) {
        const dstOffset = r * ENV_RAY_FEATURES_DIM;
        const srcOffset = r * ENV_RAY_BUFFER;

        const rRayDirX = envRaysBuffer[srcOffset + 0];
        const rRayDirY = envRaysBuffer[srcOffset + 1];
        const hitType = envRaysBuffer[srcOffset + 2];
        const rX = envRaysBuffer[srcOffset + 3];
        const rY = envRaysBuffer[srcOffset + 4];
        const rRadius = envRaysBuffer[srcOffset + 5];
        const rDistance = envRaysBuffer[srcOffset + 6];

        // Store hit type for embedding (0=none, 1=obstacle, 2=vehicle)
        envRaysTypes[r] = hitType;

        // rayDir is always present (even for misses)
        const [locRayDirX, locRayDirY] = rotateVector(rRayDirX, rRayDirY, invRotation);
        envRaysFeatures[dstOffset + 0] = locRayDirX;
        envRaysFeatures[dstOffset + 1] = locRayDirY;
        envRaysFeatures[dstOffset + 5] = norm(rDistance, QUANT);

        if (hitType !== 0) {
            const [locX, locY] = rotateVector(rX - tankX, rY - tankY, invRotation);
            envRaysFeatures[dstOffset + 2] = norm(locX, QUANT);
            envRaysFeatures[dstOffset + 3] = norm(locY, QUANT);
            envRaysFeatures[dstOffset + 4] = norm(rRadius, QUANT);
        }
    }

    // ---- Turret rays features ----
    // Format: [rayDirX, rayDirY, hitType, x, y, vx, vy, radius, distance, aimingError] -> features: [locRayDirX, locRayDirY, locX, locY, locVx, locVy, radius, distance, aimingError] + hitType for embedding
    const turretRaysFeatures = new Float32Array(TURRET_RAY_SLOTS * TURRET_RAY_FEATURES_DIM);
    const turretRaysTypes = new Int32Array(TURRET_RAY_SLOTS);
    const turretRaysBuffer = TankInputTensor.turretRaysData.getBatch(tankEid);

    for (let r = 0; r < TURRET_RAY_SLOTS; r++) {
        const dstOffset = r * TURRET_RAY_FEATURES_DIM;
        const srcOffset = r * TURRET_RAY_BUFFER;

        const tRayDirX = turretRaysBuffer[srcOffset + 0];
        const tRayDirY = turretRaysBuffer[srcOffset + 1];
        const hitType = turretRaysBuffer[srcOffset + 2];
        const tX = turretRaysBuffer[srcOffset + 3];
        const tY = turretRaysBuffer[srcOffset + 4];
        const tVx = turretRaysBuffer[srcOffset + 5];
        const tVy = turretRaysBuffer[srcOffset + 6];
        const tRadius = turretRaysBuffer[srcOffset + 7];
        const tDistance = turretRaysBuffer[srcOffset + 8];
        const tAimingError = turretRaysBuffer[srcOffset + 9];

        // Store hit type for embedding
        turretRaysTypes[r] = hitType;

        // rayDir is always present (even for misses)
        const [locRayDirX, locRayDirY] = rotateVector(tRayDirX, tRayDirY, invRotation);
        turretRaysFeatures[dstOffset + 0] = locRayDirX;
        turretRaysFeatures[dstOffset + 1] = locRayDirY;
        turretRaysFeatures[dstOffset + 7] = norm(tDistance, QUANT);

        if (hitType !== 0) {
            const [locX, locY] = rotateVector(tX - tankX, tY - tankY, invRotation);
            const [locVx, locVy] = rotateVector(tVx - speedX, tVy - speedY, invRotation);
            turretRaysFeatures[dstOffset + 2] = norm(locX, QUANT);
            turretRaysFeatures[dstOffset + 3] = norm(locY, QUANT);
            turretRaysFeatures[dstOffset + 4] = norm(locVx, QUANT);
            turretRaysFeatures[dstOffset + 5] = norm(locVy, QUANT);
            turretRaysFeatures[dstOffset + 6] = norm(tRadius, QUANT);
            turretRaysFeatures[dstOffset + 8] = tAimingError;
        }
    }

    const result = {
        battleFeatures,
        tankFeatures,
        tankType,
        enemiesFeatures,
        enemiesTypes,
        enemiesMask,
        alliesFeatures,
        alliesTypes,
        alliesMask,
        bulletsFeatures,
        bulletsMask,
        envRaysTypes,
        envRaysFeatures,
        turretRaysTypes,
        turretRaysFeatures,
    };

    if (!checkInputArrays(result)) {
        throw new Error('Invalid input arrays');
    }

    return result;
}

export function prepareRandomInputArrays(): InputArrays {
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM).map(() => random());
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM).map(() => random());
    const tankType = new Int32Array(1).map(() => randomRangeInt(0, 5));
    const enemiesMask = new Float32Array(ENEMY_SLOTS).map(() => randomRangeInt(0, 1));
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM).map(() => random());
    const enemiesTypes = new Int32Array(ENEMY_SLOTS).map(() => randomRangeInt(0, 5));
    const alliesMask = new Float32Array(ALLY_SLOTS).map(() => randomRangeInt(0, 1));
    const alliesFeatures = new Float32Array(ALLY_SLOTS * ALLY_FEATURES_DIM).map(() => random());
    const alliesTypes = new Int32Array(ALLY_SLOTS).map(() => randomRangeInt(0, 5));
    const bulletsMask = new Float32Array(BULLET_SLOTS).map(() => randomRangeInt(0, 1));
    const bulletsFeatures = new Float32Array(BULLET_SLOTS * BULLET_FEATURES_DIM).map(() => random());
    const envRaysFeatures = new Float32Array(ENV_RAY_SLOTS * ENV_RAY_FEATURES_DIM).map(() => random());
    const envRaysTypes = new Int32Array(ENV_RAY_SLOTS).map(() => randomRangeInt(0, 2));
    const turretRaysFeatures = new Float32Array(TURRET_RAY_SLOTS * TURRET_RAY_FEATURES_DIM).map(() => random());
    const turretRaysTypes = new Int32Array(TURRET_RAY_SLOTS).map(() => randomRangeInt(0, 2));

    return {
        battleFeatures,
        tankFeatures,
        tankType,
        enemiesFeatures,
        enemiesTypes,
        enemiesMask,
        alliesFeatures,
        alliesTypes,
        alliesMask,
        bulletsFeatures,
        bulletsMask,
        envRaysTypes,
        envRaysFeatures,
        turretRaysTypes,
        turretRaysFeatures,
    };
}

export function checkInputArrays(inputArray: InputArrays): boolean {
    return inputArray.battleFeatures.every(Number.isFinite)
        && inputArray.tankFeatures.every(Number.isFinite)
        && inputArray.tankType.every(Number.isFinite)
        && inputArray.enemiesFeatures.every(Number.isFinite)
        && inputArray.enemiesTypes.every(Number.isFinite)
        && inputArray.enemiesMask.every(Number.isFinite)
        && inputArray.alliesFeatures.every(Number.isFinite)
        && inputArray.alliesTypes.every(Number.isFinite)
        && inputArray.alliesMask.every(Number.isFinite)
        && inputArray.bulletsFeatures.every(Number.isFinite)
        && inputArray.bulletsMask.every(Number.isFinite)
        && inputArray.envRaysFeatures.every(Number.isFinite)
        && inputArray.envRaysTypes.every(Number.isFinite)
        && inputArray.turretRaysFeatures.every(Number.isFinite)
        && inputArray.turretRaysTypes.every(Number.isFinite);
}