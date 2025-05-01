import { GameDI } from '../../../DI/GameDI.ts';
import { TANK_RADIUS } from '../consts.ts';
import { randomRangeFloat } from '../../../../../../lib/random.ts';
import { createTank } from '../../../ECS/Entities/Tank/CreateTank.ts';
import { createPlayer } from '../../../ECS/Entities/Player.ts';

export function addRandomTanks(teamIdAndCount: [number, number][]) {
    const tanks = [];
    const width = GameDI.width;
    const height = GameDI.height;

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

    for (const [teamId, count] of teamIdAndCount) {
        for (let i = 0; i < count; i++) {
            let x: number, y: number;

            // Пытаемся найти подходящую позицию
            do {
                x = randomRangeFloat(TANK_RADIUS, width - TANK_RADIUS);
                y = randomRangeFloat(TANK_RADIUS, height - TANK_RADIUS);
            } while (isTooClose(x, y));

            const tank = createTank({
                playerId: createPlayer(teamId),
                teamId,
                x,
                y,
                rotation: Math.PI * randomRangeFloat(0, 2), // Случайный поворот от 0 до 2π
                color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
            });
            tanks.push(tank);
            tankPositions.push({ x, y });
        }
    }

    return tanks;
}