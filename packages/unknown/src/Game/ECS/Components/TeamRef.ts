import { addComponent, query, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { getPhysicsWorldComponents, PhysicsWorld } from '../createPhysicsWorld.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

export const createTeamRefComponent = defineComponent((TeamRef) => {
    const id = TypedArray.u32(delegate.defaultSize);
    return {
        id,
        addComponent(world: World, eid: number, team: number) {
            addComponent(world, eid, TeamRef);
            id[eid] = team;
        },
    };
});

export function getTeamsCount(world: PhysicsWorld) {
    const { Tank, TeamRef } = getPhysicsWorldComponents(world);
    const tanks = query(world, [Tank]);
    const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
    return teamsCount.size;
}
