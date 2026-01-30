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
 * Randomized central obstacle:
 * - 33%: two buildings placed symmetrically around the diagonal center.
 * - 66%: five buildings extending from the center in one direction.
 */
export function createScenarioDiagonalWall(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);
    const buildingSpacing = 150;

    const tankOptions = {
        fieldSize: scenario.width,
        edgeMargin: 50,
        maxDeviation: PI / 6,
        count: 2,
    };
    const geometry = createDiagonalGeometry(tankOptions);

    // Create environment: randomized central wall perpendicular to diagonal
    const perpAngle = geometry.baseDiagonalAngle + PI / 2;
    const perpX = Math.cos(perpAngle);
    const perpY = Math.sin(perpAngle);
    const random = Math.random();

    if (random < 0.33) {
        // Two buildings, centered around the diagonal center, spaced farther apart
        for (const offset of [-1, 1]) {
            createBuilding({
                x: geometry.centerX + perpX * offset * buildingSpacing,
                y: geometry.centerY + perpY * offset * buildingSpacing,
            });
        }
    } else {
        // Five buildings extending from the center in one direction
        const direction = Math.random() < 0.5 ? -1 : 1;
        for (const offset of [-1, 0, 1, 2, 3, 4]) {
            createBuilding({
                x: geometry.centerX + perpX * offset * direction * buildingSpacing,
                y: geometry.centerY + perpY * offset * direction * buildingSpacing,
            });
        }
    }

    // Create agent tanks and fill with agents
    createDiagonalTeamTanks(tankOptions, geometry, 0);
    fillWithCurrentAgents(scenario);

    // Create bot tanks and fill with bots
    createDiagonalTeamTanks(tankOptions, geometry, 1);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(0));

    return scenario;
}
