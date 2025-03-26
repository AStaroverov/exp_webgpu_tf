import { NestedArray, TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { addComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { component, obs } from '../../../../../src/ECS/utils.ts';

export const TankController = component(({
    shoot: TypedArray.i8(delegate.defaultSize),
    shootCooldown: TypedArray.f32(delegate.defaultSize),

    // Control user tank
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),

    turretDir: NestedArray.f64(2, delegate.defaultSize),

    // Methods
    addComponent(eid: number) {
        addComponent(GameDI.world, eid, TankController);
    },
    shouldShoot(eid: number): boolean {
        return TankController.shoot[eid] > 0 && TankController.shootCooldown[eid] <= 0;
    },
    setShooting$: obs((eid: number, v: boolean): void => {
        TankController.shoot[eid] = v ? 1 : 0;
    }),
    startCooldown: ((eid: number): void => {
        TankController.shootCooldown[eid] = 100;
    }),
    updateCooldown: ((eid: number, dt: number): void => {
        TankController.shootCooldown[eid] -= dt;
    }),
    setMove$: obs((eid: number, dir: number): void => {
        TankController.move[eid] = dir;
    }),
    setRotate$: obs((eid: number, dir: number): void => {
        TankController.rotation[eid] = dir;
    }),
    setTurretDir$: obs((eid: number, x: number, y: number): void => {
        TankController.turretDir.set(eid, 0, x);
        TankController.turretDir.set(eid, 1, y);
    }),
}));
