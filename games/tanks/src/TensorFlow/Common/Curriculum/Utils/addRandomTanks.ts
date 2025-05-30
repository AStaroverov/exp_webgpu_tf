import { GameDI } from '../../../../Game/DI/GameDI.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { createTank } from '../../../../Game/ECS/Entities/Tank/CreateTank.ts';
import { createPlayer } from '../../../../Game/ECS/Entities/Player.ts';
import { PI, pow, sqrt } from '../../../../../../../lib/math.ts';
import { TANK_APPROXIMATE_COLLIDER_RADIUS } from '../../../../Game/ECS/Components/HeuristicsData.ts';

export function addRandomTanks(teamIdAndCount: [number, number][]) {
    const tanks = [];
    const width = GameDI.width;
    const height = GameDI.height;

    let tankPositions: { x: number, y: number }[] = [];

    const isTooClose = (x: number, y: number): boolean => {
        for (let i = 0; i < tankPositions.length; i++) {
            const tank = tankPositions[i];
            const dist = sqrt(pow(tank.x - x, 2) + pow(tank.y - y, 2));
            if (dist < TANK_APPROXIMATE_COLLIDER_RADIUS * 2) {
                return true;
            }
        }
        return false;
    };

    for (const [teamId, count] of teamIdAndCount) {
        for (let i = 0; i < count; i++) {
            let x: number, y: number;

            do {
                x = randomRangeFloat(TANK_APPROXIMATE_COLLIDER_RADIUS, width - TANK_APPROXIMATE_COLLIDER_RADIUS);
                y = randomRangeFloat(TANK_APPROXIMATE_COLLIDER_RADIUS, height - TANK_APPROXIMATE_COLLIDER_RADIUS);
            } while (isTooClose(x, y));

            const tank = createTank({
                playerId: createPlayer(teamId),
                teamId,
                x,
                y,
                rotation: PI * randomRangeFloat(0, 2),
                color: [teamId, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
            });
            tanks.push(tank);
            tankPositions.push({ x, y });
        }
    }

    return tanks;
}