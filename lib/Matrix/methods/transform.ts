import { Matrix, TMatrix } from '../index';
import { isSubMatrix } from './submatrix';

export function transpose<T>(source: TMatrix<T>): TMatrix<T> {
    const target = Matrix.create<T>(source.h, source.w);

    Matrix.forEach(target, (_, x, y) => {
        Matrix.set(target, x, y, Matrix.get(source, y, x));
    });

    return target;
}

export function flipX<T>(source: TMatrix<T>): TMatrix<T> {
    const target = Matrix.create<T>(source.w, source.h);
    const last = source.w - 1;

    Matrix.forEach(source, (item, x, y) => {
        Matrix.set(target, last - x, y, item);
    });

    return target;
}

export function flipY<T>(source: TMatrix<T>): TMatrix<T> {
    const target = Matrix.create<T>(source.w, source.h);
    const last = source.h - 1;

    Matrix.forEach(source, (item, x, y) => {
        Matrix.set(target, x, last - y, item);
    });

    return target;
}

export function rotate<T>(source: TMatrix<T>, degree: number): TMatrix<T> {
    if (degree % 90 !== 0) {
        throw new Error('Rotate matrix only on %90 degree');
    }

    degree = degree % 360;

    if (degree % 360 === 0) {
        return Matrix.copy(source);
    }

    if (degree % 270 === 0) {
        return flipY(transpose(source));
    }

    if (degree % 180 === 0) {
        return flipY(flipX(source));
    }

    // if (degree % 90 === 0) {
    return flipX(transpose(source));
}

export function getAllVariants<T>(
    source: TMatrix<T>,
    isEqual: (a: T, b: T) => boolean = (a, b) => a === b,
): TMatrix<T>[] {
    const equal = isSubMatrix(isEqual);
    const result: TMatrix<T>[] = [source];

    const v0 = source;
    const v1 = flipX(v0);
    const v2 = flipY(v0);
    const v3 = rotate(v0, 90);

    const shouldFlipX = !equal(v0, v1);
    const shouldFlipY = !equal(v0, v2);
    const shouldRotate = !equal(v0, v3);

    if (shouldFlipX) {
        result.push(v1);

        if (shouldFlipY) {
            result.push(flipY(v1));
        }
    }

    if (shouldFlipY) {
        result.push(v2);

        if (shouldFlipX) {
            result.push(flipX(v2));
        }
    }

    if (v3 && shouldRotate) {
        const length = result.length;

        result.push(v3);
        for (let i = 1; i < length; i++) {
            result.push(rotate(result[i], 90));
        }
    }

    return result;
}
