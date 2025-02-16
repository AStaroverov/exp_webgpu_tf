import { TMatrix } from '../index';
import { get } from './base';
import { find } from './iterators/base';
import { Item } from './utils';
import { newVector, TVector, zeroVector } from '../utils/shape.ts';

export function isSubMatrix<A, B = A>(isEqual: (a: A, b: B) => boolean) {
    return function _isSubMatrix(
        matrix: TMatrix<A>,
        submatrix: TMatrix<B>,
        offset: TVector = zeroVector,
    ): boolean {
        if (submatrix.w > matrix.w || submatrix.h > matrix.h) {
            return false;
        }

        return (
            undefined ===
            find(submatrix, (b, x, y) => {
                const a = get(matrix, offset.x + x, offset.y + y);
                return a === undefined ? true : !isEqual(a, b);
            })
        );
    };
}

export function findSubMatrix<A, B = A>(
    isSubMatrix: (a: TMatrix<A>, b: TMatrix<B>, offset: TVector) => boolean,
) {
    return function _findSubMatrix(matrix: TMatrix<A>, submatrix: TMatrix<B>): undefined | Item<A> {
        return find(matrix, (_, x, y) => {
            return isSubMatrix(matrix, submatrix, newVector(x, y));
        });
    };
}

export function findSubMatrices<A, B = A>(
    isSubMatrix: (a: TMatrix<A>, b: TMatrix<B>, offset: TVector) => boolean,
) {
    return function _findSubMatrices(
        matrix: TMatrix<A>,
        submatrices: TMatrix<B>[],
    ): undefined | { item: Item<A>; submatrix: TMatrix<B> } {
        for (let i = 0; i < submatrices.length; i++) {
            const submatrix = submatrices[i];
            const item = find(matrix, (_, x, y) => {
                return isSubMatrix(matrix, submatrix, newVector(x, y));
            });

            if (item) {
                return { submatrix, item };
            }
        }
    };
}
