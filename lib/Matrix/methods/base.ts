import { isFunction } from 'lodash-es';

import { TMatrix, TMatrixSeed } from '../index';
import { seed } from './iterators/base';

export function create<T>(w: number, h: number, fillerOrBuffer?: TMatrixSeed<T> | T[]): TMatrix<T> {
    const buffer = Array.isArray(fillerOrBuffer) ? fillerOrBuffer : new Array(w * h).fill(null);
    const instance = { w, h, buffer };

    if (isFunction(fillerOrBuffer)) {
        seed(instance, fillerOrBuffer);
    }

    return instance;
}

export function get<T>(source: TMatrix<T>, x: number, y: number): undefined | T {
    return inside(source, x, y) ? source.buffer[x + y * source.w] : undefined;
}

export function set<T>(source: TMatrix<T>, x: number, y: number, item: T): undefined | T {
    return inside(source, x, y) ? (source.buffer[x + y * source.w] = item) : undefined;
}

export function copy<T>(source: TMatrix<T>): TMatrix<T> {
    return setSource(create(source.w, source.h), source.buffer);
}

export function setSource<T>(source: TMatrix<T>, buffer: T[]): TMatrix<T> {
    source.buffer = buffer;
    return source;
}

function inside(matrix: TMatrix, x: number, y: number) {
    return x >= 0 && x < matrix.w && y >= 0 && y < matrix.h;
}
