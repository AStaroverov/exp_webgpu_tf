import { createGame } from '../../../createGame.ts';
import { createTankRR, Tank } from '../../ECS/Components/Tank.ts';
import { random, randomRangeFloat, randomSign } from '../../../../../lib/random.ts';
import { GameDI } from '../../DI/GameDI.ts';
import { TANK_RADIUS } from './consts.ts';
import { TankController } from '../../ECS/Components/TankController.ts';
import { query } from 'bitecs';

const MAX_PADDING = 100;

export async function createBattlefield(tanksCount: number, withRender = false, withPlayer = false) {
    const game = await createGame({ width: 800, height: 800, withPlayer, withRender });
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
    for (let i = 0; i < tanksCount; i++) {
        let x: number, y: number;

        // Пытаемся найти подходящую позицию
        do {
            x = randomRangeFloat(TANK_RADIUS - padding, width - TANK_RADIUS + padding);
            y = randomRangeFloat(TANK_RADIUS - padding, height - TANK_RADIUS + padding);
        } while (isTooClose(x, y));

        const eid = createTankRR({
            x,
            y,
            rotation: Math.PI * randomRangeFloat(0, 2), // Случайный поворот от 0 до 2π
            color: [randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
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

    return { ...game, gameTick, getTanks };
}