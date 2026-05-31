import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createParentComponent = defineComponent((Parent) => {
    const id = new Float64Array(delegate.defaultSize);
    return {
        id,
        addComponent(world: World, eid: number, parentEid: number) {
            addComponent(world, eid, Parent);
            id[eid] = parentEid;
        },
    };
});
