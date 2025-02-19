import { TVector } from '../../../../lib/Matrix/utils/shape';
import { Assign } from '../../../../lib/Types';

export type UnknownTile = TVector & {
    type: Exclude<TileType, TileType.road>;
};
export type WallTile = Assign<UnknownTile, { type: TileType.wall; last: boolean }>;
export type RoadTile = Assign<UnknownTile, { type: TileType.road; last: boolean }>;

export type Tile = UnknownTile | RoadTile | WallTile;

export enum TileType {
    empty = 'empty',
    gross = 'gross',
    road = 'road',
    wood = 'wood',
    wall = 'wall',
}

export const getEmptyTile = (x: number, y: number): Tile => ({
    x,
    y,
    type: TileType.empty,
});

export const getRoadTile = (x: number, y: number): RoadTile => ({
    x,
    y,
    type: TileType.road,
    last: false,
});

export const getLastRoadTile = (x: number, y: number): RoadTile => ({
    ...getRoadTile(x, y),
    type: TileType.road,
    last: true,
});
