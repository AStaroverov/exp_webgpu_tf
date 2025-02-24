import { component, NestedArray, obs, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { addComponent } from 'bitecs';
import { DI } from '../../DI';

export const TankController = component(({
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),
    shot: TypedArray.i8(delegate.defaultSize),
    shotCooldown: TypedArray.f32(delegate.defaultSize),
    turretTarget: NestedArray.f64(2, delegate.defaultSize),

    // Methods
    addComponent(eid: number) {
        addComponent(DI.world, eid, TankController);
    },
    shouldShoot(eid: number, delta: number = 0): boolean {
        TankController.updateCooldown(eid, delta);
        return TankController.shot[eid] > 0 && TankController.shotCooldown[eid] <= 0;
    },
    setShooting: ((eid: number, v: boolean): void => {
        TankController.shot[eid] = v ? 1 : 0;
    }),
    startCooldown: ((eid: number): void => {
        TankController.shotCooldown[eid] = 60;
    }),
    updateCooldown: ((eid: number, dt: number): void => {
        TankController.shotCooldown[eid] -= dt;
    }),
    setMove$: obs((eid: number, dir: number): void => {
        TankController.move[eid] = dir;
    }),
    setRotate$: obs((eid: number, dir: number): void => {
        TankController.rotation[eid] = dir;
    }),
    setTurretTarget$: obs((eid: number, x: number, y: number): void => {
        TankController.turretTarget.set(eid, 0, x);
        TankController.turretTarget.set(eid, 1, y);
    }),
    getTurretTarget: (eid: number): Float64Array => {
        return TankController.turretTarget.getBatche(eid);
    },
}));
