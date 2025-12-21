// All this Zindex is just a baseline/min value for the entities.
// Zindex is used to sort the entities for rendering and shadow mapping.
export enum ZIndex {
    Background = 0,
    TreadMark = 0.001,
    Rock = 1, // more related with terrain/size
    TankHull = 4,
    TankCaterpillar = 4,
    TankTurret = 8,
    Shield = 6,
    Bullet = 8,
    // w/o shadow mapping
    Explosion = 100,
    HitFlash = 100,
    MuzzleFlash = 100,
}