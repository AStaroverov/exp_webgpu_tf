import { addComponent, World } from 'bitecs';
import { delegate } from '../../../../../src/delegate.ts';
import { component } from '../../../../../src/ECS/utils.ts';

export const Parent = component({
    id: new Float64Array(delegate.defaultSize),

    addComponent(world: World, eid: number, parentEid: number): void {
        addComponent(world, eid, Parent);
        Parent.id[eid] = parentEid;
    },
});
