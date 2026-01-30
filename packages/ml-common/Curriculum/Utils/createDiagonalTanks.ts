import { PI } from '../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../lib/random.ts';
import { createPlayer } from '../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../../tanks/src/Game/Config/vehicles.ts';

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

// Minimum distance between tanks to prevent overlap (based on HeavyTank colliderRadius = 150)
const MIN_TANK_SPACING = 200;

export type DiagonalTanksOptions = {
    fieldSize: number;
    edgeMargin: number;
    maxDeviation: number;
    spawnRadius?: number;
    count?: number; // Number of tanks per team
};

export type DiagonalTanksResult = {
    baseDiagonalAngle: number;
    centerX: number;
    centerY: number;
};

/**
 * Creates geometry info for diagonal tank placement.
 */
export function createDiagonalGeometry(options: DiagonalTanksOptions): DiagonalTanksResult {
    const { fieldSize } = options;
    return {
        baseDiagonalAngle: randomRangeFloat(0, 2 * PI),
        centerX: fieldSize / 2,
        centerY: fieldSize / 2,
    };
}

/**
 * Creates tanks for one team positioned diagonally from center.
 */
export function createDiagonalTeamTanks(
    options: DiagonalTanksOptions,
    geometry: DiagonalTanksResult,
    teamId: number,
): void {
    const { fieldSize, edgeMargin, maxDeviation, count = 1 } = options;
    const { centerX, centerY, baseDiagonalAngle } = geometry;
    const rawSpawnRadius = options.spawnRadius ?? fieldSize * 0.4;
    const maxSpawnRadius = fieldSize / 2 - edgeMargin;
    const spawnRadius = Math.max(0, Math.min(rawSpawnRadius, maxSpawnRadius));
    const baseAngle = teamId === 0 ? baseDiagonalAngle : baseDiagonalAngle + PI;
    const placedPositions: Array<{ x: number; y: number }> = [];
    const minSpacingSq = MIN_TANK_SPACING * MIN_TANK_SPACING;

    // Calculate angular step to ensure minimum spacing between tanks
    const angularStep = count > 1 ? MIN_TANK_SPACING / spawnRadius : 0;
    const totalAngularSpread = angularStep * (count - 1);
    const startAngleOffset = -totalAngularSpread / 2;

    for (let i = 0; i < count; i++) {
        const playerId = createPlayer(teamId);
        const tankAngularPosition = startAngleOffset + i * angularStep;
        const maxAttempts = 24;
        let x = 0;
        let y = 0;
        let angle = baseAngle + tankAngularPosition;
        let radius = spawnRadius;
        let placed = false;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const deviation = randomRangeFloat(-maxDeviation, maxDeviation);
            const angularJitter = count > 1 ? randomRangeFloat(-angularStep * 0.3, angularStep * 0.3) : 0;
            const radialJitter = randomRangeFloat(-20, 20);
            angle = baseAngle + deviation + tankAngularPosition + angularJitter;
            radius = Math.max(0, Math.min(maxSpawnRadius, spawnRadius + radialJitter));

            x = centerX + Math.cos(angle) * radius;
            y = centerY + Math.sin(angle) * radius;

            if (x < edgeMargin || x > fieldSize - edgeMargin || y < edgeMargin || y > fieldSize - edgeMargin) {
                continue;
            }

            const tooClose = placedPositions.some((pos) => {
                const dx = pos.x - x;
                const dy = pos.y - y;
                return dx * dx + dy * dy < minSpacingSq;
            });
            if (!tooClose) {
                placed = true;
                break;
            }
        }

        if (!placed) {
            angle = baseAngle + tankAngularPosition;
            radius = spawnRadius;
            x = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle) * radius));
            y = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle) * radius));
        }

        placedPositions.push({ x, y });

        createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId,
            teamId,
            x,
            y,
            rotation: angle + PI,
            color: teamId === 0
                ? [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1]
                : [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
    }
}
