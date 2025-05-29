import { GameDI } from '../../../../Game/DI/GameDI.ts';
import { randomRangeFloat } from '../../../../../../../lib/random.ts';
import { createMediumTank } from '../../../../Game/ECS/Entities/Tank/Medium/MediumTank.ts';
import { createPlayer } from '../../../../Game/ECS/Entities/Player.ts';
import { min, PI, pow, sqrt } from '../../../../../../../lib/math.ts';

const MIN_RADIUS = 150;

export function addRandomTanks(teamIdAndCount: [number, number][]) {
    const tanks = [];
    const width = GameDI.width;
    const height = GameDI.height;

    let tankPositions: { x: number, y: number }[] = [];

    const getMinDist = (x: number, y: number): number => {
        let minDist = Infinity;
        for (let i = 0; i < tankPositions.length; i++) {
            const tank = tankPositions[i];
            const dist = sqrt(pow(tank.x - x, 2) + pow(tank.y - y, 2));
            minDist = min(minDist, dist);
        }
        return minDist;
    };
    const findSpawnPosition = () => {
        let x: number, y: number, dist = Infinity, j = 0;

        do {
            j++;
            const rx = randomRangeFloat(50, width - 50);
            const ry = randomRangeFloat(50, height - 50);
            const d = getMinDist(rx, ry);
            if (dist === Infinity || d > dist) {
                dist = d;
                x = rx;
                y = ry;
            }
        } while (dist < MIN_RADIUS * 2 && j < 100);

        if (j >= 100) {
            console.warn('[addRandomTanks] Could not find spawn position');
        }

        return { x: x!, y: y! };
    };

    for (const [teamId, count] of teamIdAndCount) {
        for (let i = 0; i < count; i++) {
            const { x, y } = findSpawnPosition();
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