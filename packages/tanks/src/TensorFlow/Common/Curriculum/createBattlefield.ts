import { TenserFlowDI } from '../../../Game/DI/TenserFlowDI.ts';
import { createGame } from '../../../Game/createGame.ts';
import { query } from 'bitecs';
import { Tank } from '../../../Game/ECS/Components/Tank.ts';
import { TeamRef } from '../../../Game/ECS/Components/TeamRef.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

export function createBattlefield(options: { size?: number, withPlayer: boolean }) {
    TenserFlowDI.enabled = true;

    const size = options?.size ?? randomRangeInt(1200, 2000);
    const game = createGame({ width: size, height: size, ...options });

    const getTankEids = () => {
        return [...query(game.world, [Tank])];
    };

    const getTeamsCount = () => {
        const tanks = getTankEids();
        const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
        return teamsCount.size;
    };

    return { ...game, getTankEids, getTeamsCount };
}

