import { Game } from './createGame.ts';
import { query } from 'bitecs';
import { Tank } from './ECS/Components/Tank.ts';
import { TeamRef } from './ECS/Components/TeamRef.ts';
import { createTank } from './ECS/Entities/Tank/CreateTank.ts';
import { createPlayer } from './ECS/Entities/Player.ts';
import { PI } from '../../../../lib/math.ts';
import { randomRangeFloat } from '../../../../lib/random.ts';
import { GameDI } from './DI/GameDI.ts';
import { CurrentActorAgent, TankAgent } from '../TensorFlow/Common/Curriculum/Agents/CurrentActorAgent.ts';

export type GameAPI = ReturnType<typeof createGameAPI>;

export function createGameAPI(game: Game) {
    const maxTanks = 5;
    const mapTankIdToAgent = new Map<number, TankAgent>();

    const addTank = (index: number, teamId = 0) => {
        const x = GameDI.width * 0.2 + (teamId === 1 ? GameDI.width * 0.6 : 0);
        const dy = (GameDI.height - GameDI.height * 0.4) / (maxTanks - 1);
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
    const getTankEids = () => {
        return [...query(game.world, [Tank])];
    };

    const getTeamsCount = () => {
        const tanks = getTankEids();
        const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
        return teamsCount.size;
    };

    const startGame = async () => {
        const playerTeam = getTankEids();

        for (let i = 0; i < playerTeam.length; i++) {
            addTank(i, 1);
        }

        const allTanks = getTankEids();

        allTanks.forEach((tankId) => {
            mapTankIdToAgent.set(tankId, new CurrentActorAgent(tankId));
        });

        await Promise.all(Array.from(mapTankIdToAgent.values()).map(agent => agent.sync?.()));
    };

    return {
        maxTanks,
        addTank,
        getTankEids,
        getTeamsCount,
        startGame,
        destroy: () => {
            game.destroy();
            mapTankIdToAgent.forEach(agent => agent.dispose?.());
            mapTankIdToAgent.clear();
        },
    };
}
