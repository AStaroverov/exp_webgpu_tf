import { TMatrix } from '../index';
import { Point } from '../shapes/shape';
import { get } from './base';

export const STOP_ITERATE = Symbol('STOP_ITERATE');

export type Item<T> = Point & {
    value: undefined | T;
    matrix: TMatrix<T>;
};
export type ExistedItem<T> = Point & {
    value: T;
    matrix: TMatrix<T>;
};

export function createGetItem<T>(
    matrix: TMatrix<T>,
    sx: number,
    sy: number,
): (dx: number, dy: number) => Item<T> {
    return function _getItem(dx: number, dy: number): Item<T> {
        return getItem(matrix, sx + dx, sy + dy);
    };
}

export function getItem<T>(
    matrix: TMatrix<T>,
    x: number,
    y: number,
    value = get(matrix, x, y),
): Item<T> {
    return { value, x, y, matrix };
}

export const TMP_ITEM: Item<any> = {
    matrix: undefined as unknown as TMatrix<any>,
    value: undefined,
    x: 0,
    y: 0,
};
