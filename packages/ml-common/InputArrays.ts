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
} from '../tanks/src/Pilots/Components/TankState.ts';

function norm(v: number, size: number): number {
    return v / size;
}
function logNorm(v: number, size: number): number {
    return sign(v) * log1p(abs(v / size));
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

    const invRotation = -rotation;
    // const [locSpeedX, locSpeedY] = rotateVector(speedX, speedY, invRotation);
    
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
        const turretRel = turretRotation - rotation;
        
        const [locX, locY] = rotateVector(turretX - tankX, turretY - tankY, invRotation);
        turretFeatures[dstOffset + 0] = norm(locX, QUANT);
        turretFeatures[dstOffset + 1] = norm(locY, QUANT);
        turretFeatures[dstOffset + 2] = cos(turretRel); // turret rotation cos
        turretFeatures[dstOffset + 3] = sin(turretRel); // turret rotation sin
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

        const [locX, locY] = rotateVector(eX - tankX, eY - tankY, invRotation);
        const [locVx, locVy] = rotateVector(eVx - speedX, eVy - speedY, invRotation);
        const eTurretRel = eTurretRotation - rotation;

        enemiesMask[w] = 1;
        enemiesTypes[w] = eType;
        enemiesFeatures[dstOffset + 0] = hp;
        enemiesFeatures[dstOffset + 1] = norm(locX, QUANT);
        enemiesFeatures[dstOffset + 2] = norm(locY, QUANT);
        enemiesFeatures[dstOffset + 3] = norm(locVx, QUANT);
        enemiesFeatures[dstOffset + 4] = norm(locVy, QUANT);
        enemiesFeatures[dstOffset + 5] = cos(eTurretRel);
        enemiesFeatures[dstOffset + 6] = sin(eTurretRel);
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

        const [locX, locY] = rotateVector(aX - tankX, aY - tankY, invRotation);
        const [locVx, locVy] = rotateVector(aVx - speedX, aVy - speedY, invRotation);
        const aTurretRel = aTurretRotation - rotation;

        alliesMask[w] = 1;
        alliesTypes[w] = aType;
        alliesFeatures[dstOffset + 0] = hp;
        alliesFeatures[dstOffset + 1] = norm(locX, QUANT);
        alliesFeatures[dstOffset + 2] = norm(locY, QUANT);
        alliesFeatures[dstOffset + 3] = norm(locVx, QUANT);
        alliesFeatures[dstOffset + 4] = norm(locVy, QUANT);
        alliesFeatures[dstOffset + 5] = cos(aTurretRel);
        alliesFeatures[dstOffset + 6] = sin(aTurretRel);
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

        const [locX, locY] = rotateVector(bX - tankX, bY - tankY, invRotation);
        const [locSpeedX, locSpeedY] = rotateVector(bSpeedX - speedX, bSpeedY - speedY, invRotation);

        bulletsMask[w] = 1;
        bulletsFeatures[dstOffset + 0] = norm(locX, QUANT);
        bulletsFeatures[dstOffset + 1] = norm(locY, QUANT);
        bulletsFeatures[dstOffset + 2] = norm(locSpeedX, QUANT);
        bulletsFeatures[dstOffset + 3] = norm(locSpeedY, QUANT);
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
            tankX, tankY, invRotation,
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
    invRotation: number,
): void {
    const hitType = raysBuffer[srcOffset + 0];
    const rootX = raysBuffer[srcOffset + 2];
    const rootY = raysBuffer[srcOffset + 3];
    const dirX = raysBuffer[srcOffset + 4];
    const dirY = raysBuffer[srcOffset + 5];
    const distance = raysBuffer[srcOffset + 6];
    const [locRootX, locRootY] = rotateVector(rootX - tankX, rootY - tankY, invRotation);
    const [locDirX, locDirY] = rotateVector(dirX, dirY, invRotation);
    
    raysFeatures[dstOffset + 0] = norm(locRootX, QUANT);
    raysFeatures[dstOffset + 1] = norm(locRootY, QUANT);
    raysFeatures[dstOffset + 2] = locDirX;
    raysFeatures[dstOffset + 3] = locDirY;
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
