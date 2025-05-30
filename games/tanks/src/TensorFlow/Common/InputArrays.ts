import { shuffle } from '../../../../../lib/shuffle.ts';
import { ALLY_BUFFER, BULLET_BUFFER, ENEMY_BUFFER, TankInputTensor } from '../../Game/ECS/Components/TankState.ts';
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
} from '../Models/Create.ts';
import { max } from '../../../../../lib/math.ts';
import { random, randomRangeInt } from '../../../../../lib/random.ts';
import { MAX_APPROXIMATE_COLLIDER_RADIUS } from '../../Game/ECS/Components/HeuristicsData.ts';

function norm(v: number, size: number): number {
    return v / size;
}

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
    // ---- Battlefield features ----
    const battleFeatures = new Float32Array(BATTLE_FEATURES_DIM);
    let bi = 0;

    const maxCount = max(1, TankInputTensor.alliesCount[tankEid], TankInputTensor.enemiesCount[tankEid]);

    battleFeatures[bi++] = Math.log1p(width);
    battleFeatures[bi++] = Math.log1p(height);
    battleFeatures[bi++] = TankInputTensor.alliesCount[tankEid] / maxCount;
    battleFeatures[bi++] = TankInputTensor.alliesTotalHealth[tankEid] / max(1, TankInputTensor.alliesCount[tankEid]);
    battleFeatures[bi++] = TankInputTensor.enemiesCount[tankEid] / maxCount;
    battleFeatures[bi++] = TankInputTensor.enemiesTotalHealth[tankEid] / max(1, TankInputTensor.enemiesCount[tankEid]);

    // ---- Controller features ----
    const controllerFeatures = new Float32Array(CONTROLLER_FEATURES_DIM);
    let ci = 0;

    controllerFeatures[ci++] = norm(TankInputTensor.shoot[tankEid], 1);
    controllerFeatures[ci++] = norm(TankInputTensor.move[tankEid], 1);
    controllerFeatures[ci++] = norm(TankInputTensor.rotate[tankEid], 1);
    controllerFeatures[ci++] = norm(TankInputTensor.turretDir.get(tankEid, 0), 2);
    controllerFeatures[ci++] = norm(TankInputTensor.turretDir.get(tankEid, 1), 2);

    // ---- Tank features ----
    const tankFeatures = new Float32Array(TANK_FEATURES_DIM);
    let ti = 0;

    const tankX = TankInputTensor.position.get(tankEid, 0);
    const tankY = TankInputTensor.position.get(tankEid, 1);
    const rotation = TankInputTensor.rotation[tankEid];
    const speedX = TankInputTensor.speed.get(tankEid, 0);
    const speedY = TankInputTensor.speed.get(tankEid, 1);
    const turretTargetX = TankInputTensor.turretTarget.get(tankEid, 0);
    const turretTargetY = TankInputTensor.turretTarget.get(tankEid, 1);
    const colliderRadius = TankInputTensor.colliderRadius[tankEid];

    tankFeatures[ti++] = TankInputTensor.health[tankEid];
    tankFeatures[ti++] = norm(tankX - width / 2, width / 2);
    tankFeatures[ti++] = norm(tankY - height / 2, height / 2);
    tankFeatures[ti++] = norm(rotation, Math.PI);
    tankFeatures[ti++] = norm(speedX, width);
    tankFeatures[ti++] = norm(speedY, height);
    tankFeatures[ti++] = norm(turretTargetX - tankX, width);
    tankFeatures[ti++] = norm(turretTargetY - tankY, height);
    tankFeatures[ti++] = norm(colliderRadius, MAX_APPROXIMATE_COLLIDER_RADIUS);

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

        enemiesMask[w] = 1;
        enemiesFeatures[dstOffset + 0] = enemiesBuffer[srcOffset + 1];
        enemiesFeatures[dstOffset + 1] = norm(enemiesBuffer[srcOffset + 2] - tankX, width);
        enemiesFeatures[dstOffset + 2] = norm(enemiesBuffer[srcOffset + 3] - tankY, height);
        enemiesFeatures[dstOffset + 3] = norm(enemiesBuffer[srcOffset + 4], width);
        enemiesFeatures[dstOffset + 4] = norm(enemiesBuffer[srcOffset + 5], height);
        enemiesFeatures[dstOffset + 5] = norm(enemiesBuffer[srcOffset + 6] - tankX, width);
        enemiesFeatures[dstOffset + 6] = norm(enemiesBuffer[srcOffset + 7] - tankY, height);
        enemiesFeatures[dstOffset + 7] = norm(enemiesBuffer[srcOffset + 8], MAX_APPROXIMATE_COLLIDER_RADIUS);
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

        alliesMask[w] = 1;
        alliesFeatures[dstOffset + 0] = alliesBuffer[srcOffset + 1]; // hp
        alliesFeatures[dstOffset + 1] = norm(alliesBuffer[srcOffset + 2] - tankX, width);
        alliesFeatures[dstOffset + 2] = norm(alliesBuffer[srcOffset + 3] - tankY, height);
        alliesFeatures[dstOffset + 3] = norm(alliesBuffer[srcOffset + 4], width);
        alliesFeatures[dstOffset + 4] = norm(alliesBuffer[srcOffset + 5], height);
        alliesFeatures[dstOffset + 5] = norm(alliesBuffer[srcOffset + 6] - tankX, width);
        alliesFeatures[dstOffset + 6] = norm(alliesBuffer[srcOffset + 7] - tankY, height);
        alliesFeatures[dstOffset + 7] = norm(alliesBuffer[srcOffset + 8], MAX_APPROXIMATE_COLLIDER_RADIUS);
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

        bulletsMask[w] = 1;
        bulletsFeatures[dstOffset + 0] = norm(bulletsBuffer[srcOffset + 1] - tankX, width);
        bulletsFeatures[dstOffset + 1] = norm(bulletsBuffer[srcOffset + 2] - tankY, height);
        bulletsFeatures[dstOffset + 2] = norm(bulletsBuffer[srcOffset + 3], width);
        bulletsFeatures[dstOffset + 3] = norm(bulletsBuffer[srcOffset + 4], height);
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