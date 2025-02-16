import { TMatrix } from '../index';
import { forEach, reduce } from './iterators/base';
import { getItem, Item } from './utils';

export function toArray<T>(source: TMatrix<T>): T[] {
    return source.buffer;
}

export function toNestedArray<T>(matrix: TMatrix<T>): T[][] {
    const m = new Array(matrix.h).fill(null).map(() => new Array(matrix.w).fill(null));

    forEach(matrix, (v, x, y) => {
        m[y][x] = v;
    });

    return m;
}

export function toItemsArray<T>(matrix: TMatrix<T>): Item<T>[] {
    return reduce(matrix, [] as Item<T>[], (acc, _, x, y) => {
        acc.push(getItem(matrix, x, y));
        return acc;
    });
}
