import { createGame } from '../../../createGame.ts';
import { createTankRR } from '../../ECS/Components/Tank.ts';
import { random, randomRangeFloat } from '../../../../../lib/random.ts';
import { DI } from '../../DI';
import { getDrawState } from './utils.ts';
import { TANK_RADIUS } from './consts.ts';
import { TankController } from '../../ECS/Components/TankController.ts';

export function createBattlefield(tanksCount: number) {
    const game = createGame();
    const width = DI.canvas.offsetWidth;
    const height = DI.canvas.offsetHeight;
    let tanks: number[] = [];

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
    for (let i = 0; i < tanksCount; i++) {
        let x: number, y: number;

        // Пытаемся найти подходящую позицию
        do {
            x = randomRangeFloat(TANK_RADIUS, width - TANK_RADIUS);
            y = randomRangeFloat(TANK_RADIUS, height - TANK_RADIUS);
        } while (isTooClose(x, y));

        const eid = createTankRR({
            x: x,
            y: y,
            rotation: Math.PI * randomRangeFloat(0, 2), // Случайный поворот от 0 до 2π
            color: [random(), random(), random(), 1],
        });
        TankController.setTurretTarget$(eid, random() * width, random() * height);

        // Сохраняем ID и координаты танка
        tanks.push(eid);
        tankPositions.push({ x, y });
    }

    const gameTick = (delta: number) => {
        game.gameTick(delta, getDrawState());
    };

    return { ...game, tanks, gameTick };
}