import { TenserFlowDI } from '../../../Game/DI/TenserFlowDI.ts';
import { createGame } from '../../../Game/createGame.ts';
import { query } from 'bitecs';
import { Tank } from '../../../Game/ECS/Components/Tank.ts';
import { TeamRef } from '../../../Game/ECS/Components/TeamRef.ts';

export async function createBattlefield(options: { withRender: boolean, withPlayer: boolean }) {
    TenserFlowDI.enabled = true;

    const game = await createGame({ width: 1400, height: 1200, ...options });

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

