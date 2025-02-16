import { some as arraySome } from 'lodash-es';

import { TMatrix } from '../index';
import { create, get, set } from './base';
import { every, forEach, many, some } from './iterators/base';
import { shuffleMany, shuffleSome } from './iterators/shuffle';

export type ItemMatchReplace<T> = {
    match: (item: T, x: number, y: number, matrix: TMatrix<T>) => boolean;
    replace?: (item: T, x: number, y: number, matrix: TMatrix<T>, matchedMatrix: TMatrix<T>) => T;
};

export function createMatchReplace(matrixIterator = some, matchReplacesIterator = arraySome) {
    return function matchReplace<T>(
        matrix: TMatrix<T>,
        matchReplaces: TMatrix<ItemMatchReplace<T>>[],
    ): boolean {
        return matrixIterator(matrix, (_, x, y) => {
            return matchReplacesIterator(matchReplaces, createSomeReplaced(matrix, x, y));
        });
    };
}

export const matchReplace: <T>(
    matrix: TMatrix<T>,
    matchReplaces: TMatrix<ItemMatchReplace<T>>[],
) => boolean = createMatchReplace();

export const matchReplaceAll: <T>(
    matrix: TMatrix<T>,
    matchReplaces: TMatrix<ItemMatchReplace<T>>[],
) => boolean = createMatchReplace(many);

export const matchReplaceShuffle: <T>(
    matrix: TMatrix<T>,
    matchReplaces: TMatrix<ItemMatchReplace<T>>[],
) => boolean = createMatchReplace(shuffleSome);

export const matchReplaceShuffleAll: <T>(
    matrix: TMatrix<T>,
    matchReplaces: TMatrix<ItemMatchReplace<T>>[],
) => boolean = createMatchReplace(shuffleMany);

export function createSomeReplaced<T>(matrix: TMatrix<T>, sx: number, sy: number) {
    return function someReplaced(matchReplace: TMatrix<ItemMatchReplace<T>>): boolean {
        if (sx + matchReplace.w > matrix.w || sy + matchReplace.h > matrix.h) {
            return false;
        }

        const matchedItems: T[] = [];
        const matched = every(matchReplace, ({ match }, x, y, i) => {
            return match(
                (matchedItems[i] = get(matrix, sx + x, sy + y) as T),
                sx + x,
                sy + y,
                matrix,
            );
        });

        if (matched) {
            const matchedMatrix = create<T>(
                matchReplace.w,
                matchReplace.h,
                (_x, _y, i) => matchedItems[i]!,
            );
            forEach(matchReplace, ({ replace }, x, y, i) => {
                set(
                    matrix,
                    sx + x,
                    sy + y,
                    replace === undefined
                        ? matchedItems[i]
                        : replace(matchedItems[i], sx + x, sy + y, matrix, matchedMatrix),
                );
            });

            return true;
        }

        return false;
    };
}
