import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { component, obs } from '../../../../../renderer/src/ECS/utils.ts';

export const VehicleController = component(({
    move: TypedArray.f64(delegate.defaultSize),
    rotation: TypedArray.f64(delegate.defaultSize),

    // Methods
    addComponent(world: World, eid: number) {
        addComponent(world, eid, VehicleController);
        VehicleController.move[eid] = 0;
        VehicleController.rotation[eid] = 0;
    },
    setMove$: obs((eid: number, dir: number): void => {
        VehicleController.move[eid] = dir;
    }),
    setRotate$: obs((eid: number, dir: number): void => {
        VehicleController.rotation[eid] = dir;
    }),
}));

