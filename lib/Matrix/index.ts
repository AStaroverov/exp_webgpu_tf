import { copy, create, get, set, setSource } from './methods/base';
import { fromNestedArray } from './methods/from';
import { every, fill, find, forEach, map, reduce, seed, slice, some } from './methods/iterators/base';
import { match, matchAll, matchShuffle, matchShuffleAll } from './methods/match';
import { matchReplace, matchReplaceAll, matchReplaceShuffle, matchReplaceShuffleAll } from './methods/matchReplace';
import { getSide } from './methods/side';
import { findSubMatrices, findSubMatrix, isSubMatrix } from './methods/submatrix';
import { toArray, toItemsArray, toNestedArray } from './methods/to';
import { flipX, flipY, getAllVariants, rotate, transpose } from './methods/transform';
import { STOP_ITERATE } from './methods/utils';

export type TMatrix<T = unknown> = { w: number, h: number, buffer: T[] };
export type TMatrixSeed<T> = (x: number, y: number, i: number) => T;

export const Matrix = {
    create,
    seed,
    setSource,

    STOP_ITERATE,
    slice,
    forEach,
    reduce,
    find,
    some,
    every,
    fill,
    map,
    get,
    set,
    copy,

    fromNestedArray,

    toArray,
    toItemsArray,
    toNestedArray,

    transpose,
    flipX,
    flipY,
    rotate,

    isSubMatrix,
    findSubMatrix,
    findSubMatrices,

    getSide,

    getAllVariants,

    match,
    matchAll,
    matchShuffle,
    matchShuffleAll,

    matchReplace,
    matchReplaceAll,
    matchReplaceShuffle,
    matchReplaceShuffleAll,
};
