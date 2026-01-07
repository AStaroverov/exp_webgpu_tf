import { PI } from '../../../lib/math.ts';
import { randomRangeFloat } from '../../../lib/random.ts';
import { createBuilding } from '../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { createDiagonalTanks } from './Utils/createDiagonalTanks.ts';
import { add1v1Pilots } from './Utils/add1v1Pilots.ts';
import { Scenario } from './types.ts';

/**
 * Diagonal scenario: 1 agent vs 1 simple bot positioned diagonally from center.
 * A building is placed in the center as an objective/obstacle.
 */
export function createScenarioDiagonal(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);

    // Create tanks
    const { agentTankEid, botTankEid, centerX, centerY } = createDiagonalTanks({
        fieldSize: scenario.width,
        edgeMargin: 300,
        maxDeviation: PI / 12,
    });

    // Create environment
    createBuilding({ x: centerX, y: centerY });

    // Add pilots
    add1v1Pilots(scenario, agentTankEid, botTankEid, {
        move: randomRangeFloat(0, 0.05),
        aim: {
            aimError: randomRangeFloat(0.6, 0.9),
            shootChance: randomRangeFloat(0.01, 0.1),
        },
    });

    return scenario;
}
