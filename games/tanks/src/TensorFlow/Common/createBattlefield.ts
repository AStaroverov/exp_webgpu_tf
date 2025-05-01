import { createGame } from '../../createGame.ts';
import { Tank } from '../../ECS/Components/Tank.ts';
import { random, randomRangeFloat, randomRangeInt } from '../../../../../lib/random.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { TANK_RADIUS } from './consts.ts';
import { query } from 'bitecs';
import { TenserFlowDI } from '../../DI/TenserFlowDI.ts';
import { TeamRef } from '../../ECS/Components/TeamRef.ts';
import { createTank } from '../../ECS/Entities/Tank/CreateTank.ts';
import { createPlayer } from '../../ECS/Entities/Player.ts';

const MAX_PADDING = 0;

export async function createBattlefield(tanksCount: number, withRender = false, withPlayer = false) {
    TenserFlowDI.enabled = true;

    const game = await createGame({ width: 1400, height: 1200, withPlayer, withRender });
    const width = GameDI.width;
    const height = GameDI.height;
    const padding = random() * MAX_PADDING;

    // Храним координаты танков в отдельном массиве
    let tankPositions: { x: number, y: number }[] = [];

    // Функция для проверки минимального расстояния между танками
    const isTooClose = (x: number, y: number): boolean => {
        for (let i = 0; i < tankPositions.length; i++) {
            const tank = tankPositions[i];
            const dist = Math.sqrt(Math.pow(tank.x - x, 2) + Math.pow(tank.y - y, 2));
            if (dist < TANK_RADIUS * 2) {
                return true;
            }
        }
        return false;
    };

    for (let i = 0; i < tanksCount; i++) {
        let x: number, y: number;

        // Пытаемся найти подходящую позицию
        do {
            x = randomRangeFloat(TANK_RADIUS - padding, width - TANK_RADIUS + padding);
            y = randomRangeFloat(TANK_RADIUS - padding, height - TANK_RADIUS + padding);
        } while (isTooClose(x, y));

        const teamId = i % 2;
        createTank({
            playerId: createPlayer(teamId),
            teamId,
            x,
            y,
            rotation: Math.PI * randomRangeFloat(0, 2), // Случайный поворот от 0 до 2π
            color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        tankPositions.push({ x, y });
    }

    const gameTick = (delta: number) => {
        game.gameTick(delta);
    };

    const getTanks = () => {
        return [...query(GameDI.world, [Tank])];
    };

    const getTeamsCount = () => {
        const tanks = getTanks();
        const teamsCount = new Set(tanks.map(tankId => TeamRef.id[tankId]));
        return teamsCount.size;
    };

    const activeAgents = new Set([getTanks()[randomRangeInt(0, tanksCount - 1)]]); // , ...getTanks().filter(() => random() > 0.85)
    const getAgenTanks = () => {
        return getTanks().filter((id) => activeAgents.has(id));
    };

    return { ...game, gameTick, getTanks, getAgenTanks, getTeamsCount };
}