export const PLAYER_REFS = {
    tankPid: null as null | number,
};

export enum ZIndex {
    Background = 0,
    TankHull = 0.001,
    TankCaterpillar = 0.001,
    TankTurret = 0.0011,
    Bullet = 0.002,
    Explosion = 0.003,
}