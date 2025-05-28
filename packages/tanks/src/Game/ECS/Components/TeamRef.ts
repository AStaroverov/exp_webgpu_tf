import { component } from '../../../../../renderer/src/ECS/utils.ts';
import { TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, query, World } from 'bitecs';
import { Tank } from './Tank.ts';
import { GameDI } from '../../DI/GameDI.ts';

export const TeamRef = component({
    id: TypedArray.u32(delegate.defaultSize),

    addComponent: (world: World, eid: number, team: number) => {
        addComponent(world, eid, TeamRef);
        TeamRef.id[eid] = team;
    },
});

export function getTeamsCount({ world } = GameDI) {
    const tanks = query(world, [Tank]);
    const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
    return teamsCount.size;
}