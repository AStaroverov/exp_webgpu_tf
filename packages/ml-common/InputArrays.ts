import { random, randomRangeInt } from '../../lib/random.ts';
import { shuffle } from '../../lib/shuffle.ts';
import {
    ALLY_FEATURES_DIM,
    ALLY_SLOTS,
    BATTLE_FEATURES_DIM,
    BULLET_FEATURES_DIM,
    BULLET_SLOTS,
    CONTROLLER_FEATURES_DIM,
    ENEMY_FEATURES_DIM,
    ENEMY_SLOTS,
    TANK_FEATURES_DIM,
} from '../ml/src/Models/Create.ts';
import { ALLY_BUFFER, BULLET_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../tanks/src/Pilots/Components/TankState.ts';

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
    controllerFeatures: Float32Array,
    battleFeatures: Float32Array,
    tankFeatures: Float32Array,
    enemiesFeatures: Float32Array,
    enemiesMask: Float32Array,
    alliesFeatures: Float32Array,
    alliesMask: Float32Array,
    bulletsFeatures: Float32Array
    bulletsMask: Float32Array,
}

export function prepareInputArrays(
    tankEid: number,
    width: number,
    height: number,
): InputArrays {
    // ---- ????? features ----
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM);
    let bi = 0;

    battleFeatures[bi++] = Math.log1p(width);
    battleFeatures[bi++] = Math.log1p(height);
    battleFeatures[bi++] = 0;
    battleFeatures[bi++] = 0;
    battleFeatures[bi++] = 0;
    battleFeatures[bi++] = 0;

    // ---- ????? features ----
    const controllerFeatures = new Float32Array(CONTROLLER_FEATURES_DIM);
    let ci = 0;

    controllerFeatures[ci++] = 0;
    controllerFeatures[ci++] = 0;
    controllerFeatures[ci++] = 0;
    controllerFeatures[ci++] = 0;

    // ---- Tank features ----
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM);
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
    tankFeatures[ti++] = Math.sin(rotation); // body rotation sin
    tankFeatures[ti++] = Math.cos(rotation); // body rotation cos
    tankFeatures[ti++] = norm(locSpeedX, QUANT); // speedX
    tankFeatures[ti++] = norm(locSpeedY, QUANT); // speedY
    tankFeatures[ti++] = Math.sin(turretRel); // turret rotation sin
    tankFeatures[ti++] = Math.cos(turretRel); // turret rotation cos
    tankFeatures[ti++] = norm(colliderRadius, QUANT); // collider radius

    // ---- Enemies features ----
    const enemiesMask = new Float32Array(ENEMY_SLOTS);
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

        const eX = enemiesBuffer[srcOffset + 2];
        const eY = enemiesBuffer[srcOffset + 3];
        const eRot = enemiesBuffer[srcOffset + 4];
        const eSpeedX = enemiesBuffer[srcOffset + 5];
        const eSpeedY = enemiesBuffer[srcOffset + 6];
        const eTurretRot = enemiesBuffer[srcOffset + 7];
        const eColliderRadius = enemiesBuffer[srcOffset + 8];

        const [locX, locY] = rotateVector(eX - tankX, eY - tankY, invRotation);
        const [locSpeedX, locSpeedY] = rotateVector(eSpeedX - speedX, eSpeedY - speedY, invRotation);
        const relRot = eRot - rotation;
        const relTurretRot = eTurretRot - rotation;

        enemiesMask[w] = 1;
        enemiesFeatures[dstOffset + 0] = enemiesBuffer[srcOffset + 1]; // hp
        enemiesFeatures[dstOffset + 1] = norm(locX, QUANT); // x
        enemiesFeatures[dstOffset + 2] = norm(locY, QUANT); // y
        enemiesFeatures[dstOffset + 3] = Math.sin(relRot); // body rotation sin
        enemiesFeatures[dstOffset + 4] = Math.cos(relRot); // body rotation cos
        enemiesFeatures[dstOffset + 5] = norm(locSpeedX, QUANT); // speedX
        enemiesFeatures[dstOffset + 6] = norm(locSpeedY, QUANT); // speedY
        enemiesFeatures[dstOffset + 7] = Math.sin(relTurretRot); // turret rotation sin
        enemiesFeatures[dstOffset + 8] = Math.cos(relTurretRot); // turret rotation cos
        enemiesFeatures[dstOffset + 9] = norm(eColliderRadius, QUANT); // collider radius
    }

    // ---- Allies features ----
    const alliesMask = new Float32Array(ALLY_SLOTS);
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

        const aX = alliesBuffer[srcOffset + 2];
        const aY = alliesBuffer[srcOffset + 3];
        const aRot = alliesBuffer[srcOffset + 4];
        const aSpeedX = alliesBuffer[srcOffset + 5];
        const aSpeedY = alliesBuffer[srcOffset + 6];
        const aTurretRot = alliesBuffer[srcOffset + 7];
        const aColliderRadius = alliesBuffer[srcOffset + 8];

        const [locX, locY] = rotateVector(aX - tankX, aY - tankY, invRotation);
        const [locSpeedX, locSpeedY] = rotateVector(aSpeedX - speedX, aSpeedY - speedY, invRotation);
        const relRot = aRot - rotation;
        const relTurretRot = aTurretRot - rotation;

        alliesMask[w] = 1;
        alliesFeatures[dstOffset + 0] = alliesBuffer[srcOffset + 1]; // hp
        alliesFeatures[dstOffset + 1] = norm(locX, QUANT);
        alliesFeatures[dstOffset + 2] = norm(locY, QUANT);
        alliesFeatures[dstOffset + 3] = Math.sin(relRot);
        alliesFeatures[dstOffset + 4] = Math.cos(relRot);
        alliesFeatures[dstOffset + 5] = norm(locSpeedX, QUANT);
        alliesFeatures[dstOffset + 6] = norm(locSpeedY, QUANT);
        alliesFeatures[dstOffset + 7] = Math.sin(relTurretRot);
        alliesFeatures[dstOffset + 8] = Math.cos(relTurretRot);
        alliesFeatures[dstOffset + 9] = norm(aColliderRadius, QUANT);
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

    const result = {
        controllerFeatures,
        battleFeatures,
        tankFeatures,
        enemiesFeatures,
        enemiesMask,
        alliesFeatures,
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
    const controllerFeatures = new Float32Array(CONTROLLER_FEATURES_DIM).map(() => random());
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM).map(() => random());
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM).map(() => random());
    const enemiesMask = new Float32Array(ENEMY_SLOTS).map(() => randomRangeInt(0, 1));
    const enemiesFeatures = new Float32Array(ENEMY_SLOTS * ENEMY_FEATURES_DIM).map(() => random());
    const alliesMask = new Float32Array(ALLY_SLOTS).map(() => randomRangeInt(0, 1));
    const alliesFeatures = new Float32Array(ALLY_SLOTS * ALLY_FEATURES_DIM).map(() => random());
    const bulletsMask = new Float32Array(BULLET_SLOTS).map(() => randomRangeInt(0, 1));
    const bulletsFeatures = new Float32Array(BULLET_SLOTS * BULLET_FEATURES_DIM).map(() => random());

    return {
        controllerFeatures,
        battleFeatures,
        tankFeatures,
        enemiesFeatures,
        enemiesMask,
        alliesFeatures,
        alliesMask,
        bulletsFeatures,
        bulletsMask,
    };
}

export function checkInputArrays(inputArray: InputArrays): boolean {
    return inputArray.controllerFeatures.every(Number.isFinite)
        && inputArray.battleFeatures.every(Number.isFinite)
        && inputArray.tankFeatures.every(Number.isFinite)
        && inputArray.enemiesFeatures.every(Number.isFinite)
        && inputArray.enemiesMask.every(Number.isFinite)
        && inputArray.alliesFeatures.every(Number.isFinite)
        && inputArray.alliesMask.every(Number.isFinite)
        && inputArray.bulletsFeatures.every(Number.isFinite)
        && inputArray.bulletsMask.every(Number.isFinite);
}