import { TMatrix } from '../../index';
import { createGetItem, Item } from '../utils';
import { abs, floor, max } from '../../../math.ts';
import { mapVector, newVector, setVector, sumVector, TVector } from '../../utils/shape.ts';

export function* lineIterate<T>(
    matrix: TMatrix<T>,
    start: TVector,
    direction: TVector,
): IterableIterator<Item<T>> {
    const getItem = createGetItem(matrix, start.x, start.y);
    const passed = newVector(0, 0);
    const step = newVector(
        direction.x / max(abs(direction.x), abs(direction.y)),
        direction.y / max(abs(direction.x), abs(direction.y)),
    );

    yield getItem(passed.x, passed.y);

    while (passed.x !== direction.x || passed.y !== direction.y) {
        setVector(passed, sumVector(passed, step));

        const coord = mapVector(passed, floor);

        yield getItem(coord.x, coord.y);
    }
}
