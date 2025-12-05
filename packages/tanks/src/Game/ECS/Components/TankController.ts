import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../../../../../renderer/src/ECS/utils.ts';

export const TankController = component(({
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),

    shoot: TypedArray.f32(delegate.defaultSize),
    turretRotation: TypedArray.f64(delegate.defaultSize),

    // Methods
    addComponent(world: World, eid: number) {
        addComponent(world, eid, TankController);
        TankController.shoot[eid] = 0;
        TankController.move[eid] = 0;
        TankController.rotation[eid] = 0;
        TankController.turretRotation[eid] = 0;
    },
    shouldShoot(eid: number): boolean {
        return TankController.shoot[eid] > 0;
    },
    setShooting$: obs((eid: number, v: number): void => {
        TankController.shoot[eid] = v;
    }),
    setMove$: obs((eid: number, dir: number): void => {
        TankController.move[eid] = dir;
    }),
    setRotate$: obs((eid: number, dir: number): void => {
        TankController.rotation[eid] = dir;
    }),
    setTurretRotation$: obs((eid: number, v: number): void => {
        TankController.turretRotation[eid] = v;
    }),
}));
