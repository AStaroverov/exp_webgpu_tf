import { addComponent, removeComponent, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createTeamRefComponent = defineComponent((TeamRef) => {
    const id = TypedArray.u32(delegate.defaultSize);
    return {
        id,
        addComponent(world: World, eid: number, team: number) {
            addComponent(world, eid, TeamRef);
            id[eid] = team;
        },
        removeComponent(world: World, eid: number) {
            id[eid] = 0;
            removeComponent(world, eid, TeamRef);
        },
    };
});
