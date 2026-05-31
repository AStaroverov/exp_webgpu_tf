import { TypedArray } from '../../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../../renderer/src/ECS/utils.ts';

/** Parameters for a MoveToHex action. */
export const createMoveToHexParamsComponent = defineComponent((MoveToHexParams) => {
    const speed = TypedArray.f64(delegate.defaultSize);
    return {
        speed,
        addComponent(world: World, eid: number, s: number = 1) {
            addComponent(world, eid, MoveToHexParams);
            speed[eid] = s;
        },
    };
});
