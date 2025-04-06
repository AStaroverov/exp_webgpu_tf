// import { GameDI } from '../../DI/GameDI.ts';
// import { getEmptyTile, TileType } from '../../TilesMatrix/def';
// import { Matrix } from '../../../../../lib/Matrix';
// import { fillEnvironment } from '../../TilesMatrix/fillers/environment';
// import { fillWalls } from '../../TilesMatrix/fillers/walls.ts';
// import { Cuboid } from '@dimforge/rapier2d-simd/rapier';
// import { once } from 'lodash-es';
//
// export function createMapSystem({ physicalWorld } = GameDI) {
//     return once(() => {
//         const matrix = Matrix.create(100, 100, getEmptyTile);
//         fillEnvironment(matrix);
//         fillWalls(matrix);
//
//         Matrix.forEach(matrix, (item, x, y) => {
//             if (item.type === TileType.wall) {
//                 const options = {
//                     x: x * 10,
//                     y: y * 10,
//                     width: 10,
//                     height: 10,
//                 };
//                 const intersected = null !== physicalWorld.intersectionWithShape(
//                     options, 0, new Cuboid(10 * options.width / 2, 10 * options.height / 2),
//                 );
//                 // !intersected && createWallRR(options);
//             }
//         });
//     });
// }