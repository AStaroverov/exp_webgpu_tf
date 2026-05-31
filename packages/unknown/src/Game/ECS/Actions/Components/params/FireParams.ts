import { TypedArray } from '../../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../../renderer/src/ECS/utils.ts';

/** Parameters for a Fire action. `shots` = number of rounds to fire before finishing. */
export const createFireParamsComponent = defineComponent((FireParams) => {
    const shots = TypedArray.f64(delegate.defaultSize);
    return {
        shots,
        addComponent(world: World, eid: number, n: number = 1) {
            addComponent(world, eid, FireParams);
            shots[eid] = n;
        },
    };
});
