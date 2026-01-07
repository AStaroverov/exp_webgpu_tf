import { PI } from '../../../lib/math.ts';
import { createBuilding } from '../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { createDiagonalTanks } from './Utils/createDiagonalTanks.ts';
import { add1v1Pilots } from './Utils/add1v1Pilots.ts';
import { Scenario } from './types.ts';

/**
 * Diagonal scenario with wall: 1 agent vs 1 simple bot positioned diagonally.
 * 3 buildings are placed perpendicular to the diagonal (forming a wall between tanks).
 */
export function createScenarioDiagonalWall(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);
    const buildingSpacing = 80;

    // Create tanks
    const { agentTankEid, botTankEid, baseDiagonalAngle, centerX, centerY } = createDiagonalTanks({
        fieldSize: scenario.width,
        edgeMargin: 50,
        maxDeviation: PI / 6,
    });

    // Create environment: wall of 3 buildings perpendicular to diagonal
    const perpAngle = baseDiagonalAngle + PI / 2;
    for (const offset of [-1, 0, 1]) {
        createBuilding({
            x: centerX + Math.cos(perpAngle) * offset * buildingSpacing,
            y: centerY + Math.sin(perpAngle) * offset * buildingSpacing,
        });
    }

    // Add pilots
    add1v1Pilots(scenario, agentTankEid, botTankEid);

    return scenario;
}
