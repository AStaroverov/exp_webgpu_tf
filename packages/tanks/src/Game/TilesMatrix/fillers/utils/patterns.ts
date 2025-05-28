import {
    isEmptyTile,
    isGrossTile,
    isNotEmptyTile,
    isNotGrossTile,
    isNotLastRoadTile,
    isNotRoadTile,
    isNotWallTile,
    isRoadTile,
    isWallTile,
} from './is.ts';

export const matchEmpty = {
    match: isEmptyTile,
};

export const matchNotEmpty = {
    match: isNotEmptyTile,
};

export const matchGross = {
    match: isGrossTile,
};

export const matchNotGross = {
    match: isNotGrossTile,
};

export const matchRoad = {
    match: isRoadTile,
};

export const matchNotLastRoad = {
    match: isNotLastRoadTile,
};

export const matchNotRoad = {
    match: isNotRoadTile,
};

export const matchWall = {
    match: isWallTile,
};

export const matchNotWall = {
    match: isNotWallTile,
};
