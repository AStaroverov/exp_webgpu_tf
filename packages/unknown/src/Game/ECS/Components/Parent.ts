import { addComponent, hasComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { getGameComponents } from '../createGameWorld.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createParentComponent = defineComponent((Parent) => {
    const id = new Float64Array(delegate.defaultSize);
    return {
        id,
        addComponent(world: World, eid: number, parentEid: number) {
            addComponent(world, eid, Parent);
            id[eid] = parentEid;

            const { Children } = getGameComponents(world);
            if (!hasComponent(world, parentEid, Children)) {
                console.warn('Parent component added to entity without Children component');
            }
        },
    };
});
