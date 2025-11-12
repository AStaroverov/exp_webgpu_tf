import { Matrix, TMatrix } from '../../../../../../lib/Matrix';
import { random } from '../../../../../../lib/random.ts';
import { Tile, TileType, WallTile } from '../def.ts';
import { isNotEmptyTile, isNotWallTile, isWallTile, isWallTileNotLast } from './utils/is.ts';
import { matchNotWall } from './utils/patterns.ts';

const createReplaceToWall = (chance: number) => (tile: Tile) => {
    return <WallTile>Object.assign(tile, {
        tankType: TileType.wall,
        last: random() > chance,
    });
};

const tryReplaceToWall = (chanceOnReplace: number, chanceOnLast: number) => {
    const replaceToWall = createReplaceToWall(1 - chanceOnLast);

    return (tile: Tile) => {
        return Math.random() < chanceOnReplace ? replaceToWall(tile) : tile;
    };
};

const matchWall = {
    match: isWallTile,
};

const matchNotLastWall = {
    match: isWallTileNotLast,
};

const matchEmptyTryReplaceToWall = {
    match: isNotEmptyTile,
    replace: tryReplaceToWall(0.01, 0),
};

const matchNotWallReplaceToWall = {
    match: isNotWallTile,
    replace: createReplaceToWall(0.8),
};

const randomReplaceToWall = [(
    Matrix.fromNestedArray([
        /* eslint-disable */
        [matchEmptyTryReplaceToWall],
        /* eslint-enable */
    ])
)];

const continueWallPattern = Matrix.getAllVariants(
    Matrix.fromNestedArray([
        /* eslint-disable */
        [matchNotWall, matchNotWall],
        [matchNotWall, matchNotWall],
        [matchNotLastWall, matchNotWallReplaceToWall],
        // [matchWall, matchNotWallReplaceToWall],
        [matchNotWall, matchNotWall],
        [matchNotWall, matchNotWall],
        /* eslint-enable */
    ]),
);

const rotateWallPattern = Matrix.getAllVariants(
    Matrix.fromNestedArray([
        /* eslint-disable */
        [matchNotWall, matchNotWall, matchNotWall],
        [matchWall, matchNotLastWall, matchNotWall],
        [matchNotWall, matchNotWallReplaceToWall, matchNotWall],
        /* eslint-enable */
    ]),
);


const boldWallPattern = Matrix.getAllVariants(
    Matrix.fromNestedArray([
        /* eslint-disable */
        [matchNotWall],
        [matchWall],
        [matchNotWallReplaceToWall],
        [matchNotWall],
        /* eslint-enable */
    ]),
);

const filterRotate = () => random() > 0.6;
const getRotatePattern = () => rotateWallPattern.filter(filterRotate);

export function fillWalls(matrix: TMatrix<Tile>) {
    Matrix.matchReplaceShuffleAll(matrix, randomReplaceToWall);

    while (true) {
        const step1 = random() > 0 && Matrix.matchReplace(matrix, getRotatePattern());
        const step2 = Matrix.matchReplaceAll(matrix, continueWallPattern);

        if (!(step1 || step2)) {
            break;
        }
    }

    Matrix.matchReplaceAll(matrix, boldWallPattern);
}
