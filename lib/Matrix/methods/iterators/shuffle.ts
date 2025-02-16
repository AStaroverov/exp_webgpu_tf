import { TMatrix } from '../../index';
import { get } from '../base';
import { STOP_ITERATE } from '../utils';
import { createEvery, createFind, createMany, createMap, createReduce, createSome } from './create';
import { shuffle } from '../../../shuffle.ts';

const NUMBERS = new Int16Array(1e3).map((_, i) => i);
const SHUFFLED_NUMBERS_1 = new Int16Array(1e3).map((_, i) => i);
const SHUFFLED_NUMBERS_2 = new Int16Array(1e3).map((_, i) => i);

export function shuffleForEach<T>(
    matrix: TMatrix<T>,
    callback: (item: T, x: number, y: number, i: number) => unknown,
): boolean {
    SHUFFLED_NUMBERS_1.set(NUMBERS);
    SHUFFLED_NUMBERS_2.set(NUMBERS);

    const w = matrix.w;
    const h = matrix.h;
    const xs = shuffle(SHUFFLED_NUMBERS_1.subarray(0, w));
    const ys = shuffle(SHUFFLED_NUMBERS_2.subarray(0, h));

    let index = 0;
    let x = 0;
    let y = 0;

    for (let i = 0; i < w; i++) {
        x = xs[i];

        for (let j = 0; j < h; j++) {
            y = ys[j];

            if (callback(get(matrix, x, y)!, x, y, index++) === STOP_ITERATE) {
                return true;
            }
        }
    }

    return false;
}

export const shuffleReduce = createReduce(shuffleForEach);
export const shuffleFind = createFind(shuffleForEach);
export const shuffleSome = createSome(shuffleForEach);
export const shuffleMany = createMany(shuffleForEach);
export const shuffleEvery = createEvery(shuffleForEach);
export const shuffleMap = createMap(shuffleForEach);
