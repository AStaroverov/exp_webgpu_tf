import { TankAgent } from '../../../TensorFlow/Common/Curriculum/Agents/CurrentActorAgent.ts';
import { GameDI } from '../../../Game/DI/GameDI.ts';
import { createTank } from '../../../Game/ECS/Entities/Tank/CreateTank.ts';
import { createPlayer } from '../../../Game/ECS/Entities/Player.ts';
import { PI } from '../../../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { query } from 'bitecs';
import { Tank } from '../../../Game/ECS/Components/Tank.ts';
import { TeamRef } from '../../../Game/ECS/Components/TeamRef.ts';
import { getEngine } from './engine.ts';

export const GAME_MAX_TEAM_TANKS = 5;
export const mapTankIdToAgent = new Map<number, TankAgent>();

export const addTank = (index: number, teamId = 0) => {
    const x = GameDI.width * 0.2 + (teamId === 1 ? GameDI.width * 0.6 : 0);
    const dy = (GameDI.height - GameDI.height * 0.4) / (GAME_MAX_TEAM_TANKS - 1);
    const y = GameDI.height * 0.2 + index * dy;

    createTank({
        playerId: createPlayer(0),
        teamId,
        x,
        y,
        rotation: PI * randomRangeFloat(0, 2),
        color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });
};

export const getTankEids = () => {
    return [...query(getEngine().world, [Tank])];
};

export const getTeamsCount = () => {
    const tanks = getTankEids();
    const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
    return teamsCount.size;
};

