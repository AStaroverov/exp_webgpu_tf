import { createGame } from '../../../createGame.ts';
import { Tank } from '../../ECS/Components/Tank/Tank.ts';
import { random, randomRangeFloat, randomSign } from '../../../../../lib/random.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { TANK_RADIUS } from './consts.ts';
import { TankController } from '../../ECS/Components/Tank/TankController.ts';
import { query } from 'bitecs';
import { TenserFlowDI } from '../../DI/TenserFlowDI.ts';
import { Team } from '../../ECS/Components/Team.ts';
import { createTank } from '../../ECS/Components/Tank/CreateTank.ts';
import { getNewPlayerId } from '../../ECS/Components/Player.ts';

const MAX_PADDING = 100;

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

    // Создаем танки с проверкой минимального расстояния
    let teamZeroCount = Math.floor(tanksCount / 2);
    for (let i = 0; i < tanksCount; i++) {
        let x: number, y: number;

        // Пытаемся найти подходящую позицию
        do {
            x = randomRangeFloat(TANK_RADIUS - padding, width - TANK_RADIUS + padding);
            y = randomRangeFloat(TANK_RADIUS - padding, height - TANK_RADIUS + padding);
        } while (isTooClose(x, y));

        const teamId = teamZeroCount-- > 0 ? 0 : 1;
        const eid = createTank({
            playerId: getNewPlayerId(),
            teamId,
            x,
            y,
            rotation: Math.PI * randomRangeFloat(0, 2), // Случайный поворот от 0 до 2π
            color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        TankController.setTurretDir$(
            eid,
            randomSign() * random(),
            randomSign() * random(),
        );

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
        const teamsCount = new Set(tanks.map(tankId => Team.id[tankId]));
        return teamsCount.size;
    };

    return { ...game, gameTick, getTanks, getTeamsCount };
}