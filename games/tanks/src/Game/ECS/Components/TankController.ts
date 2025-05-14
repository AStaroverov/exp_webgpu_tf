import { NestedArray, TypedArray } from '../../../../../../src/utils.ts';
import { delegate } from '../../../../../../src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../../../../../../src/ECS/utils.ts';

export const TankController = component(({
    shoot: TypedArray.f32(delegate.defaultSize),
    shootCooldown: TypedArray.f32(delegate.defaultSize),

    // Control user tank
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),

    turretDir: NestedArray.f64(2, delegate.defaultSize),

    // Methods
    addComponent(world: World, eid: number) {
        addComponent(world, eid, TankController);
        TankController.shoot[eid] = 0;
        TankController.shootCooldown[eid] = 0;
        TankController.move[eid] = 0;
        TankController.rotation[eid] = 0;
        TankController.turretDir.set(eid, 0, 0);
        TankController.turretDir.set(eid, 1, 0);
    },
    shouldShoot(eid: number): boolean {
        return TankController.shoot[eid] > 0 && TankController.shootCooldown[eid] <= 0;
    },
    setShooting$: obs((eid: number, v: number): void => {
        TankController.shoot[eid] = v;
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
