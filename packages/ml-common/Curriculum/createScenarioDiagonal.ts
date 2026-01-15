import { PI } from '../../../lib/math.ts';
import { randomRangeFloat } from '../../../lib/random.ts';
import { createBuilding } from '../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { createDiagonalGeometry, createDiagonalTeamTanks } from './Utils/createDiagonalTanks.ts';
import { fillWithCurrentAgents } from './Utils/fillWithCurrentAgents.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';
import { Scenario } from './types.ts';

/**
 * Diagonal scenario: 1 agent vs 1 simple bot positioned diagonally from center.
 * A building is placed in the center as an objective/obstacle.
 */
export function createScenarioDiagonal(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);

    const tankOptions = {
        fieldSize: scenario.width,
        edgeMargin: 300,
        maxDeviation: PI / 12,
        count: 1,
    };
    const geometry = createDiagonalGeometry(tankOptions);

    // Create environment
    createBuilding({ x: geometry.centerX, y: geometry.centerY });

    // Create agent tanks and fill with agents
    createDiagonalTeamTanks(tankOptions, geometry, 0);
    fillWithCurrentAgents(scenario);

    // Create bot tanks and fill with bots
    createDiagonalTeamTanks(tankOptions, geometry, 1);
    fillWithSimpleHeuristicAgents(scenario, {
        move: randomRangeFloat(0, 0.05),
        aim: {
            aimError: randomRangeFloat(0.6, 0.9),
            shootChance: randomRangeFloat(0.01, 0.1),
        },
    });

    return scenario;
}
