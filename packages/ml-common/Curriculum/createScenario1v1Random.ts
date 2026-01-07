import { query } from 'bitecs';
import { PI } from '../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../lib/random.ts';
import { Vehicle } from '../../tanks/src/Game/ECS/Components/Vehicle.ts';
import { getTankTeamId } from '../../tanks/src/Game/ECS/Entities/Tank/TankUtils.ts';
import { CurrentActorAgent } from '../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { SimpleBot } from '../../tanks/src/Pilots/Agents/SimpleBot.ts';
import { createPilotsPlugin } from '../../tanks/src/Pilots/createPilotsPlugin.ts';
import { Pilot } from '../../tanks/src/Pilots/Components/Pilot.ts';
import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { getSuccessRatio as computeSuccessRatio, getTeamHealth } from './utils.ts';
import { getTeamsCount } from '../../tanks/src/Game/ECS/Components/TeamRef.ts';
import { resetSpawnGrid, getSpawnGrid, getCellWorldPosition, setCellContent, CellContent } from '../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createPlayer } from '../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../tanks/src/Game/Config/vehicles.ts';

/**
 * Simplest scenario: 1 agent vs 1 simple bot at random positions.
 * No fauna, no obstacles - pure 1v1 combat training.
 */
export function createScenario1v1Random(options: Parameters<typeof createBattlefield>[0] & {
    index: number;
    train?: boolean;
}): Scenario {
    const game = createBattlefield(options);
    const pilots = createPilotsPlugin(game);
    const isTrain = options.train ?? true;

    resetSpawnGrid();
    const grid = getSpawnGrid();
    const totalCells = grid.cols * grid.rows;

    // Pick two random non-overlapping cells
    const usedIndices = new Set<number>();
    const getRandomCell = () => {
        let index: number;
        do {
            index = randomRangeInt(0, totalCells - 1);
        } while (usedIndices.has(index));
        usedIndices.add(index);
        return {
            col: index % grid.cols,
            row: Math.floor(index / grid.cols),
        };
    };

    const tanks: number[] = [];
    const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

    // Create agent tank (team 0)
    {
        const { col, row } = getRandomCell();
        const { x, y } = getCellWorldPosition(col, row);
        const playerId = createPlayer(0);
        const tank = createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId,
            teamId: 0,
            x,
            y,
            rotation: PI * randomRangeFloat(0, 2),
            color: [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        setCellContent(col, row, CellContent.Vehicle, tank);
        tanks.push(tank);
    }

    // Create enemy bot (team 1)
    {
        const { col, row } = getRandomCell();
        const { x, y } = getCellWorldPosition(col, row);
        const playerId = createPlayer(1);
        const tank = createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId,
            teamId: 1,
            x,
            y,
            rotation: PI * randomRangeFloat(0, 2),
            color: [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        setCellContent(col, row, CellContent.Vehicle, tank);
        tanks.push(tank);
    }

    const activeTeam = getTankTeamId(tanks[0]);
    const initialTeamHealth = getTeamHealth(tanks);

    const getVehicleEids = () => query(game.world, [Vehicle]);
    const getSuccessRatio = () => computeSuccessRatio(activeTeam, initialTeamHealth, getTeamHealth(tanks));

    const scenario: Scenario = {
        ...game,
        ...pilots,
        index: options.index,
        isTrain,
        getVehicleEids,
        getTeamsCount,
        getSuccessRatio,
    };

    // Add agent pilot to first tank
    Pilot.addComponent(game.world, tanks[0], new CurrentActorAgent(tanks[0], isTrain));

    // Add simple bot pilot to enemy tank
    Pilot.addComponent(game.world, tanks[1], new SimpleBot(tanks[1], {
        move: randomRangeFloat(0.1, 0.3),
        aim: {
            aimError: randomRangeFloat(0.6, 0.9),
            shootChance: randomRangeFloat(0.01, 0.1),
        },
    }));

    pilots.toggle(true);

    return scenario;
}

