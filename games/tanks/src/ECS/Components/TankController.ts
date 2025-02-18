import { addComponent, defineComponent, IWorld, Types } from 'bitecs';

export const TankController = defineComponent({
    move: Types.f64,
    rotation: Types.f64,
    shot: Types.i8,
    turretTarget: [Types.f64, 2],
});

export function addTankControllerComponent(world: IWorld, eid: number): void {
    addComponent(world, TankController, eid);
}

export function setTankControllerShot(eid: number): void {
    TankController.shot[eid] = 1;
}

export function shouldTankControllerShot(eid: number): boolean {
    return TankController.shot[eid] > 0;
}

export function resetTankControllerShot(eid: number): void {
    TankController.shot[eid] = 0;
}

export function setTankControllerMove(eid: number, dir: number): void {
    TankController.move[eid] = dir;
}

export function setTankControllerRotate(eid: number, dir: number): void {
    TankController.rotation[eid] = dir;
}

export function setTankControllerEnemyTarget(eid: number, x: number, y: number): void {
    TankController.turretTarget[eid][0] = x;
    TankController.turretTarget[eid][1] = y;
}

export function getTankControllerTurretTarget(eid: number): Float64Array {
    return TankController.turretTarget[eid];
}