import { TypedArray } from '../../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../../renderer/src/ECS/utils.ts';

/** Parameters for a Wait action: a simple timer (ms). */
export const createWaitParamsComponent = defineComponent((WaitParams) => {
    const duration = TypedArray.f64(delegate.defaultSize);
    const elapsed = TypedArray.f64(delegate.defaultSize);
    return {
        duration,
        elapsed,
        addComponent(world: World, eid: number, d: number = 0) {
            addComponent(world, eid, WaitParams);
            duration[eid] = d;
            elapsed[eid] = 0;
        },
    };
});
