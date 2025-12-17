import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../../../../../renderer/src/ECS/utils.ts';

export const TurretController = component(({
    shoot: TypedArray.f32(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),

    // Methods
    addComponent(world: World, eid: number) {
        addComponent(world, eid, TurretController);
        TurretController.shoot[eid] = 0;
        TurretController.rotation[eid] = 0;
    },
    shouldShoot(eid: number): boolean {
        return TurretController.shoot[eid] > 0;
    },
    setShooting$: obs((eid: number, v: number): void => {
        TurretController.shoot[eid] = v;
    }),
    setRotation$: obs((eid: number, v: number): void => {
        TurretController.rotation[eid] = v;
    }),
}));

