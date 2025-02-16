import { TMatrix } from '../../index';
import { createGetItem, Item } from '../utils';
import { lineIterate } from './lineIterate';
import { mulVector, newVector, TVector } from '../../utils/shape.ts';

export function* crossIterate<T>(
    matrix: TMatrix<T>,
    start: TVector,
    radius: number,
): IterableIterator<Item<T>> {
    const getItem = createGetItem(matrix, start.x, start.y);
    const directions = [newVector(1, 0), newVector(-1, 0), newVector(0, 1), newVector(0, -1)].map(
        (v) => mulVector(v, radius),
    );

    yield getItem(0, 0);

    for (let i = 0; i < directions.length; i++) {
        const iterator = lineIterate(matrix, start, directions[i]);
        // skip first element
        let step = iterator.next();

        while ((step = iterator.next()).done !== true) {
            yield step.value;
        }
    }
}
