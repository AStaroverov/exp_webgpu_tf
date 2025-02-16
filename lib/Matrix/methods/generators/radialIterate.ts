import { TMatrix } from '../../index';
import { createGetItem, Item } from '../utils';
import { floor, max } from '../../../math.ts';

export function* radialIterate<T>(
    matrix: TMatrix<T>,
    sx?: number,
    sy?: number,
    radius?: number,
): IterableIterator<Item<T>> {
    sx = floor(matrix.w / 2);
    sy = floor(matrix.h / 2);
    radius = (radius ?? floor(max(matrix.w, matrix.h) / 2)) - 1;
    const getItem = createGetItem(matrix, sx, sy);

    yield getItem(0, 0);

    let dx = 0;
    let dy = 0;

    let step = 0;
    let shift = 0;
    while (step < radius) {
        step += 1;
        const size = step * 2;

        // top
        shift = 0;
        while (shift < size) {
            dx = -step + shift++;
            dy = -step;
            yield getItem(dx, dy);
        }

        // right
        shift = 0;
        while (shift < size) {
            dx = +step;
            dy = -step + shift++;
            yield getItem(dx, dy);
        }

        // bottom
        shift = 0;
        while (shift < size) {
            dx = step - shift++;
            dy = step;
            yield getItem(dx, dy);
        }

        // left
        shift = 0;
        while (shift < size) {
            dx = -step;
            dy = +step - shift++;
            yield getItem(dx, dy);
        }
    }
}
