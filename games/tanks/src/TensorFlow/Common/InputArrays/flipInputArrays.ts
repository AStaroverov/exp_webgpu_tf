import { InputArrays } from './prepareInputArrays.ts';
import { ALLY_FEATURES_DIM, BULLET_FEATURES_DIM, ENEMY_FEATURES_DIM } from '../../Models/Create.ts';

export type FlipMode = 'none' | 'x' | 'y' | 'xy';

export function flipInputArrays(input: InputArrays, m: FlipMode = 'none'): InputArrays {
    if (m === 'none') return input;

    const out: InputArrays = {
        controllerFeatures: input.controllerFeatures.slice(),
        battleFeatures: input.battleFeatures.slice(),
        tankFeatures: input.tankFeatures.slice(),
        enemiesFeatures: input.enemiesFeatures.slice(),
        enemiesMask: input.enemiesMask,
        alliesFeatures: input.alliesFeatures.slice(),
        alliesMask: input.alliesMask,
        bulletsFeatures: input.bulletsFeatures.slice(),
        bulletsMask: input.bulletsMask,
    };

    /* ---------- controller ---------- */
    {
        const cf = out.controllerFeatures;
        [cf[3], cf[4]] = flipVec(m, cf[3], cf[4]);
        if (m !== 'xy') cf[2] = -cf[2];
    }

    /* ---------- tank ---------- */
    {
        const tf = out.tankFeatures;
        [tf[1], tf[2]] = flipVec(m, tf[1], tf[2]); // position
        tf[3] = flipAngle(m, tf[3]);               // rotation
        [tf[4], tf[5]] = flipVec(m, tf[4], tf[5]); // speed
        [tf[6], tf[7]] = flipVec(m, tf[6], tf[7]); // turret position
    }

    processBatch(
        m,
        out.enemiesFeatures,
        ENEMY_FEATURES_DIM,
        [
            [1, 2], // position
            [3, 4], // speed
            [5, 6], // turret position
        ],
    );

    processBatch(
        m,
        out.alliesFeatures,
        ALLY_FEATURES_DIM,
        [
            [1, 2], // position
            [3, 4], // speed
            [5, 6], // turret position
        ],
    );

    processBatch(
        m,
        out.bulletsFeatures,
        BULLET_FEATURES_DIM,
        [
            [0, 1], // position
            [2, 3], // speed
        ],
    );

    /* ---------- battle-features (ширина/высота) ---------- */
    if (m === 'xy') {
        const bf = out.battleFeatures;              // log(width), log(height)
        [bf[0], bf[1]] = [bf[1], bf[0]];
    }

    return out;
}


export function flipVec(mode: FlipMode, x: number, y: number): [number, number] {
    switch (mode) {
        case 'x':
            return [x, -y];
        case 'y':
            return [-x, y];
        case 'xy':
            return [y, x];
        default:
            return [x, y];
    }
}

export function flipAngle(mode: FlipMode, a: number): number {
    switch (mode) {
        case 'x':
            return -a;
        case 'y':
            return Math.PI - a;
        case 'xy':
            return Math.PI / 2 - a;
        default:
            return a;
    }
}

function processBatch(
    mode: FlipMode,
    features: Float32Array,
    stride: number,
    pairsFlipVec: [number, number][],
    pairsFlipAngle: [number, number][] = [],
) {
    const slots = features.length / stride;
    for (let i = 0; i < slots; i++) {
        const base = i * stride;

        // векторы
        for (const [ix, iy] of pairsFlipVec) {
            [features[base + ix], features[base + iy]] = flipVec(mode, features[base + ix], features[base + iy]);
        }

        // углы
        for (const [ai] of pairsFlipAngle) {
            features[base + ai] = flipAngle(mode, features[base + ai]);
        }
    }
}
