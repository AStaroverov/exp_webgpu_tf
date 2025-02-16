import { TMatrix } from '../../index';
import { create, get, set } from '../base';
import { STOP_ITERATE } from '../utils';
import { createEvery, createFind, createMany, createMap, createReduce, createSome } from './create';

export function slice<T>(
    matrix: TMatrix<T>,
    sx: number,
    sy: number,
    w: number,
    h: number,
): TMatrix<T> {
    return create<T>(w, h, (x, y): T => {
        return get(matrix, sx + x, sy + y) as T;
    });
}

export function forEach<T>(
    matrix: TMatrix<T>,
    callback: (item: T, x: number, y: number, i: number) => unknown,
): boolean {
    const length = matrix.w * matrix.h;

    for (let i = 0; i < length; i++) {
        const x = i % matrix.w;
        const y = (i / matrix.w) | 0;

        if (callback(matrix.buffer[i], x, y, i) === STOP_ITERATE) {
            return true;
        }
    }

    return false;
}

export function seed<T>(matrix: TMatrix<T>, filler: (x: number, y: number, i: number) => T): void {
    forEach(matrix, (_, x, y, i) => {
        set(matrix, x, y, filler(x, y, i));
    });
}

export function fill<T>(matrix: TMatrix<T>, filler: (item: T, x: number, y: number) => T): void {
    forEach(matrix, (item, x, y) => {
        set(matrix, x, y, filler(item, x, y));
    });
}

export const reduce = createReduce(forEach);
export const find = createFind(forEach);
export const some = createSome(forEach);
export const many = createMany(forEach);
export const every = createEvery(forEach);
export const map = createMap(forEach);
