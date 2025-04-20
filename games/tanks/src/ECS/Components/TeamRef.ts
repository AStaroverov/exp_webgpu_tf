import { component } from '../../../../../src/ECS/utils.ts';
import { TypedArray } from '../../../../../src/utils.ts';
import { delegate } from '../../../../../src/delegate.ts';
import { addComponent, World } from 'bitecs';

export const TeamRef = component({
    id: TypedArray.u32(delegate.defaultSize),

    addComponent: (world: World, eid: number, team: number) => {
        addComponent(world, eid, TeamRef);
        TeamRef.id[eid] = team;
    },
});
