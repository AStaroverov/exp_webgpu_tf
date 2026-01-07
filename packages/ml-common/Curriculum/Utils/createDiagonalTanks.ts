import { PI } from '../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../lib/random.ts';
import { createPlayer } from '../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../../tanks/src/Game/Config/vehicles.ts';

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

export type DiagonalTanksOptions = {
    fieldSize: number;
    edgeMargin: number;
    maxDeviation: number;
    spawnRadius?: number;
};

export type DiagonalTanksResult = {
    agentTankEid: number;
    botTankEid: number;
    baseDiagonalAngle: number;
    centerX: number;
    centerY: number;
};

/**
 * Creates two tanks positioned diagonally from the center of the field.
 * Returns tank eids and geometry info for placing obstacles.
 */
export function createDiagonalTanks(options: DiagonalTanksOptions): DiagonalTanksResult {
    const { fieldSize, edgeMargin, maxDeviation } = options;
    const centerX = fieldSize / 2;
    const centerY = fieldSize / 2;
    const spawnRadius = options.spawnRadius ?? fieldSize * 0.4;

    // Random base diagonal angle (0-360 degrees)
    const baseDiagonalAngle = randomRangeFloat(0, 2 * PI);

    // Create agent tank (team 0) - with independent deviation from diagonal
    const deviation1 = randomRangeFloat(-maxDeviation, maxDeviation);
    const angle1 = baseDiagonalAngle + deviation1;
    const x1 = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle1) * spawnRadius));
    const y1 = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle1) * spawnRadius));
    const agentPlayerId = createPlayer(0);
    const rotationToCenter1 = angle1 + PI;
    const agentTankEid = createTank({
        type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
        playerId: agentPlayerId,
        teamId: 0,
        x: x1,
        y: y1,
        rotation: rotationToCenter1,
        color: [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    // Create enemy bot (team 1) - opposite side with its own independent deviation
    const deviation2 = randomRangeFloat(-maxDeviation, maxDeviation);
    const angle2 = baseDiagonalAngle + PI + deviation2;
    const x2 = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle2) * spawnRadius));
    const y2 = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle2) * spawnRadius));
    const botPlayerId = createPlayer(1);
    const rotationToCenter2 = angle2 + PI;
    const botTankEid = createTank({
        type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
        playerId: botPlayerId,
        teamId: 1,
        x: x2,
        y: y2,
        rotation: rotationToCenter2,
        color: [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    return {
        agentTankEid,
        botTankEid,
        baseDiagonalAngle,
        centerX,
        centerY,
    };
}

