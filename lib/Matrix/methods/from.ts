import { Matrix, TMatrix } from '../index';

export function fromNestedArray<T>(v: T[][]): TMatrix<T> {
    return Matrix.setSource(Matrix.create(v[0].length, v.length), v.flat());
}
