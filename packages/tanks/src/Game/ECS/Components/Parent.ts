import { addComponent, hasComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { Children } from './Children.ts';

export const Parent = component({
    id: new Float64Array(delegate.defaultSize),

    addComponent(world: World, eid: number, parentEid: number): void {
        addComponent(world, eid, Parent);
        Parent.id[eid] = parentEid;

        if (!hasComponent(world, parentEid, Children)) {
            console.warn('Parent component added to entity without Children component');
        }
    },
});
