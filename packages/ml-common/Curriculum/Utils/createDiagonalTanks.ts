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
    const spawnRadius = options.spawnRadius ?? fieldSize * 0.4;
    const baseAngle = teamId === 0 ? baseDiagonalAngle : baseDiagonalAngle + PI;

    // Calculate angular step to ensure minimum spacing between tanks
    const angularStep = count > 1 ? MIN_TANK_SPACING / spawnRadius : 0;
    const totalAngularSpread = angularStep * (count - 1);
    const startAngleOffset = -totalAngularSpread / 2;

    for (let i = 0; i < count; i++) {
        const playerId = createPlayer(teamId);
        const deviation = randomRangeFloat(-maxDeviation, maxDeviation);
        const tankAngularPosition = startAngleOffset + i * angularStep;
        const angularJitter = count > 1 ? randomRangeFloat(-angularStep * 0.3, angularStep * 0.3) : 0;
        const radialJitter = randomRangeFloat(-20, 20);

        const angle = baseAngle + deviation + tankAngularPosition + angularJitter;
        const radius = spawnRadius + radialJitter;

        const x = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle) * radius));
        const y = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle) * radius));

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
