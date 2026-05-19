import { PI } from '../../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../../lib/random.ts';
import { createPlayer } from '../../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { createBuilding } from '../../../../tanks/src/Game/ECS/Entities/Building/index.ts';
import { VehicleType } from '../../../../tanks/src/Game/Config/vehicles.ts';
import {
    resetSpawnGrid,
    getSpawnGrid,
    isCellEmpty,
    getCellWorldPosition,
    setCellContent,
    CellContent,
} from '../../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createScenarioCore, ScenarioCoreOptions } from '../createScenarioCore.ts';
import { Scenario } from '../types.ts';
import { BotLevel, createBotFeatures } from './botFeatures.ts';
import { fillWithCurrentAgents } from './fillWithCurrentAgents.ts';
import { fillWithSimpleHeuristicAgents } from './fillWithSimpleHeuristicAgents.ts';

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

/**
 * Creates a scenario with N agents and M bots at random grid positions.
 * Uses SpawnGrid to guarantee no overlap between tanks and obstacles.
 */
export function createRandomNvsMScenario(
    options: ScenarioCoreOptions,
    agentsCount: number,
    botsCount: number,
    botLevel: BotLevel = 0,
): Scenario {
    const scenario = createScenarioCore(options);
    resetSpawnGrid();
    const grid = getSpawnGrid();

    // Place one building in a random interior cell
    const interiorCells: { col: number; row: number }[] = [];
    for (let row = 1; row < grid.rows - 1; row++) {
        for (let col = 1; col < grid.cols - 1; col++) {
            interiorCells.push({ col, row });
        }
    }
    const buildingCell = interiorCells[randomRangeInt(0, interiorCells.length - 1)];
    const buildingPos = getCellWorldPosition(buildingCell.col, buildingCell.row);
    createBuilding({ x: buildingPos.x, y: buildingPos.y });
    setCellContent(buildingCell.col, buildingCell.row, CellContent.Obstacle);

    // Spawn tanks in random empty cells
    const spawnTank = (teamId: number) => {
        const totalCells = grid.cols * grid.rows;
        let attempts = 0;
        let col: number, row: number;
        do {
            const idx = randomRangeInt(0, totalCells - 1);
            col = idx % grid.cols;
            row = Math.floor(idx / grid.cols);
            attempts++;
        } while (!isCellEmpty(col!, row!) && attempts < 200);

        const { x, y } = getCellWorldPosition(col!, row!);
        setCellContent(col!, row!, CellContent.Vehicle);

        createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId: createPlayer(teamId),
            teamId,
            x,
            y,
            rotation: PI * randomRangeFloat(0, 2),
            color: teamId === 0
                ? [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1]
                : [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
    };

    for (let i = 0; i < agentsCount; i++) spawnTank(0);
    fillWithCurrentAgents(scenario);

    for (let i = 0; i < botsCount; i++) spawnTank(1);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(botLevel));

    return scenario;
}
