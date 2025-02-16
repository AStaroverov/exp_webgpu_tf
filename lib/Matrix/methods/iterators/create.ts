import { TMatrix } from '../../index';
import { create, get, set } from '../base';
import { getItem, Item, STOP_ITERATE } from '../utils';
import { forEach } from './base';
import { shuffleForEach } from './shuffle';

export function createReduce(iterate: typeof forEach | typeof shuffleForEach) {
    return function reduce<T, Acc>(
        matrix: TMatrix<T>,
        accumulator: Acc,
        callback: (accumulator: Acc, item: T, x: number, y: number, i: number) => Acc,
    ): Acc {
        function _reduce(item: T, x: number, y: number, i: number) {
            accumulator = callback(accumulator, item, x, y, i);
        }

        iterate(matrix, _reduce);

        return accumulator;
    };
}

export function createFind(iterate: typeof forEach | typeof shuffleForEach) {
    return function find<T>(
        matrix: TMatrix<T>,
        callback: (item: T, x: number, y: number, i: number) => boolean,
    ): undefined | Item<T> {
        let item;

        function _find(v: any, x: number, y: number, i: number) {
            if (callback(v, x, y, i)) {
                item = getItem(matrix, x, y, matrix.buffer[i]);
                return STOP_ITERATE;
            }
        }

        iterate(matrix, _find);

        return item;
    };
}

export function createSome(iterate: typeof forEach | typeof shuffleForEach) {
    return function some<T>(
        matrix: TMatrix<T>,
        callback: (item: T, x: number, y: number, i: number) => boolean,
    ): boolean {
        return createFind(iterate)(matrix, callback) !== undefined;
    };
}

export function createMany(iterate: typeof forEach | typeof shuffleForEach) {
    return function many<T>(
        matrix: TMatrix<T>,
        callback: (item: T, x: number, y: number, i: number) => boolean,
    ): boolean {
        let result = false;

        function _many(v: any, x: number, y: number, i: number) {
            const r = callback(v, x, y, i);
            result = result || r;
        }

        iterate(matrix, _many);

        return result;
    };
}

export function createEvery(iterate: typeof forEach | typeof shuffleForEach) {
    return function every<T>(
        matrix: TMatrix<T>,
        callback: (item: T, x: number, y: number, i: number) => boolean,
    ): boolean {
        function _find(v: any, x: number, y: number, i: number) {
            return !callback(v, x, y, i);
        }

        return createFind(iterate)(matrix, _find) === undefined;
    };
}

export function createMap(iterate: typeof forEach | typeof shuffleForEach) {
    return function map<T>(
        source: TMatrix<T>,
        mapper: (item: T, x: number, y: number) => T,
    ): TMatrix<T> {
        const target = create<T>(source.w, source.h);

        iterate(source, (_: unknown, x: number, y: number) => {
            set(target, x, y, mapper(get(source, x, y)!, x, y));
        });

        return target;
    };
}
