import { TMatrix } from '../index';
import { create, get, set } from './base';
import { every, many, some } from './iterators/base';
import { shuffleMany, shuffleSome } from './iterators/shuffle';

export type ItemMatch<T> = {
    match: (item: T, x: number, y: number, matrix: TMatrix<T>) => boolean;
};

export const match: <T>(matrix: TMatrix<T>, matches: TMatrix<ItemMatch<T>>[]) => TMatrix<T>[] =
    createMatch();

export const matchAll: <T>(matrix: TMatrix<T>, matches: TMatrix<ItemMatch<T>>[]) => TMatrix<T>[] =
    createMatch(many);

export const matchShuffle: <T>(
    matrix: TMatrix<T>,
    matches: TMatrix<ItemMatch<T>>[],
) => TMatrix<T>[] = createMatch(shuffleSome);

export const matchShuffleAll: <T>(
    matrix: TMatrix<T>,
    matches: TMatrix<ItemMatch<T>>[],
) => TMatrix<T>[] = createMatch(shuffleMany);

export function createMatch(matrixIterator = some) {
    return function match<T>(
        matrix: TMatrix<T>,
        targetMatrices: TMatrix<ItemMatch<T>>[],
    ): TMatrix<T>[] {
        const peaces: TMatrix<T>[] = [];

        matrixIterator(matrix, (_, x, y) => {
            const matcher = createMatcher(matrix, x, y);
            return targetMatrices.some((target) => {
                const peace = matcher(target);
                return peace !== undefined ? (peaces.push(peace), true) : false;
            });
        });

        return peaces;
    };
}

const SHARED_MATRIX_SOURCE = new Array(100 & 100).fill(null);
const SHARED_MATRIX = create<any>(100, 100, SHARED_MATRIX_SOURCE);

export function createMatcher<T>(matrix: TMatrix<T>, sx: number, sy: number) {
    return function matcher(targetMatrix: TMatrix<ItemMatch<T>>): undefined | TMatrix<T> {
        if (sx + targetMatrix.w > matrix.w || sy + targetMatrix.h > matrix.h) {
            return undefined;
        }

        SHARED_MATRIX.w = targetMatrix.w;
        SHARED_MATRIX.h = targetMatrix.h;

        const matched = every(targetMatrix, ({ match }, x, y) => {
            return match(
                set(SHARED_MATRIX, x, y, get(matrix, sx + x, sy + y) as T)!,
                sx + x,
                sy + y,
                matrix,
            );
        });

        return matched
            ? create<T>(
                targetMatrix.w,
                targetMatrix.h,
                SHARED_MATRIX_SOURCE.slice(0, targetMatrix.w * targetMatrix.h),
            )
            : undefined;
    };
}
