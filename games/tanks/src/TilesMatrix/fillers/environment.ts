import pipe from 'lodash/fp/pipe';

import { Tile, TileType } from '../def';
import { isEmptyTile } from './utils/is';
import {
    getProbabilityRecord,
    getRandomProbability,
    normalizeProbabilities,
    ProbabilityRecord,
} from './utils/probabilities';
import { Matrix, TMatrix } from '../../../../../lib/Matrix';

const matchEmptyReplaceToSome = {
    match: isEmptyTile,
    replace: updateTile,
};

const matchReplaceEnvironment = Matrix.getAllVariants(
    Matrix.fromNestedArray([
        /* eslint-disable */
        [matchEmptyReplaceToSome],
        /* eslint-enable */
    ]),
);

export function fillEnvironment(matrix: TMatrix<Tile>) {
    while (true) {
        const step1 = Matrix.matchReplaceAll(matrix, matchReplaceEnvironment);

        if (!step1) {
            break;
        }
    }
}

function updateTile(tile: Tile, x: number, y: number, matrix: TMatrix<Tile>): Tile {
    const slice = Matrix.slice<Tile>(matrix, x - 1, y - 1, 3, 3);

    tile.type = pipe(
        getProbabilityRecord(getTileProbabilities),
        normalizeProbabilities,
        getRandomProbability,
    )(slice) as TileType;

    return tile;
}

function getTileProbabilities(tile: undefined | Tile): undefined | ProbabilityRecord {
    if (tile === undefined) {
        return undefined;
    }

    return {
        [TileType.gross]: 1,
    };
    //
    // if (tile.type === TileType.gross) {
    //     return {
    //         [TileType.well]: 0.1,
    //         [TileType.gross]: 0.9,
    //     };
    // }
    //
    // // if (tile.type === TileType.wood) {
    // //     return {
    // //         [TileType.well]: 0.2,
    // //         [TileType.gross]: 0.15,
    // //         [TileType.wood]: 0.65,
    // //     };
    // // }
    //
    // if (tile.type === TileType.well) {
    //     return {
    //         [TileType.well]: 0.3,
    //         [TileType.gross]: 0.7,
    //     };
    // }
    //
    // return {
    //     // [TileType.empty]: 0.85,
    //     [TileType.well]: 0.2,
    //     [TileType.gross]: 0.8,
    //     // [TileType.wood]: 0.05,
    //     // [TileType.gross]: 0.05,
    // };
}
