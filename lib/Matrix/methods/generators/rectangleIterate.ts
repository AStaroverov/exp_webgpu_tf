import { Point } from '../../../shape';
import { Matrix, TMatrix } from '../../index';

export type Item<T> = Point & {
    value: T;
    matrix: TMatrix<T>;
};

export function* rectangleIterate<T>(
    matrix: TMatrix<T>,
    sx: number,
    sy: number,
    w: number,
    h: number,
): IterableIterator<undefined | Item<T>> {
    // top
    let shift = 0;
    while (shift < w) {
        const item = getItem(shift++, 0);

        if (item !== undefined) {
            yield item;
        }
    }

    // right
    shift = 0;
    while (shift < h) {
        const item = getItem(w, shift++);

        if (item !== undefined) {
            yield item;
        }
    }

    // bottom
    shift = 0;
    while (shift < h) {
        const item = getItem(w - shift++, h);

        if (item !== undefined) {
            yield item;
        }
    }

    // left
    shift = 0;
    while (shift < h) {
        const item = getItem(0, h - shift++);

        if (item !== undefined) {
            yield item;
        }
    }

    function getItem(dx: number, dy: number) {
        const value = Matrix.get(matrix, sx + dx, sy + dy);
        return value ? { value, x: sx + dx, y: sy + dy, matrix } : undefined;
    }
}
