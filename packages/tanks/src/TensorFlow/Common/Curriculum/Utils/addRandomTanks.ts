import { GameDI } from '../../../../Game/DI/GameDI.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { createMediumTank } from '../../../../Game/ECS/Entities/Tank/Medium/MediumTank.ts';
import { createPlayer } from '../../../../Game/ECS/Entities/Player.ts';
import { PI, pow, sqrt } from '../../../../../../../lib/math.ts';

const MIN_RADIUS = 150;

export function addRandomTanks(teamIdAndCount: [number, number][]) {
    const tanks = [];
    const width = GameDI.width;
    const height = GameDI.height;

    let tankPositions: { x: number, y: number }[] = [];

    const isTooClose = (x: number, y: number): boolean => {
        for (let i = 0; i < tankPositions.length; i++) {
            const tank = tankPositions[i];
            const dist = sqrt(pow(tank.x - x, 2) + pow(tank.y - y, 2));
            if (dist < MIN_RADIUS * 2) {
                return true;
            }
        }
        return false;
    };

    for (const [teamId, count] of teamIdAndCount) {
        for (let i = 0; i < count; i++) {
            let x: number, y: number;

            do {
                x = randomRangeFloat(MIN_RADIUS, width - MIN_RADIUS);
                y = randomRangeFloat(MIN_RADIUS, height - MIN_RADIUS);
            } while (isTooClose(x, y));

            const tank = createMediumTank({
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