import { PI } from '../../../lib/math.ts';
import { createBuilding } from '../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { createDiagonalGeometry, createDiagonalTeamTanks } from './Utils/createDiagonalTanks.ts';
import { createBotFeatures } from './Utils/botFeatures.ts';
import { fillWithCurrentAgents } from './Utils/fillWithCurrentAgents.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';
import { Scenario } from './types.ts';

/**
 * Diagonal scenario with wall: 2 agents vs 2 simple bots positioned diagonally.
 * 3 buildings are placed perpendicular to the diagonal (forming a wall between teams).
 */
export function createScenarioDiagonalWall(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);
    const buildingSpacing = 100;

    const tankOptions = {
        fieldSize: scenario.width,
        edgeMargin: 50,
        maxDeviation: PI / 6,
        count: 2,
    };
    const geometry = createDiagonalGeometry(tankOptions);

    // Create environment: wall of 3 buildings perpendicular to diagonal
    const perpAngle = geometry.baseDiagonalAngle + PI / 2;
    for (const offset of [-1, 0, 1]) {
        createBuilding({
            x: geometry.centerX + Math.cos(perpAngle) * offset * buildingSpacing,
            y: geometry.centerY + Math.sin(perpAngle) * offset * buildingSpacing,
        });
    }

    // Create agent tanks and fill with agents
    createDiagonalTeamTanks(tankOptions, geometry, 0);
    fillWithCurrentAgents(scenario);

    // Create bot tanks and fill with bots
    createDiagonalTeamTanks(tankOptions, geometry, 1);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(0));

    return scenario;
}
