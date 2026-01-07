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
import { createPlayer } from '../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../tanks/src/Game/Config/vehicles.ts';
import { createBuilding } from '../../tanks/src/Game/ECS/Entities/Building/index.ts';

/**
 * Diagonal scenario: 1 agent vs 1 simple bot positioned diagonally from center.
 * A building is placed in the center as an objective/obstacle.
 */
export function createScenarioDiagonal(options: Parameters<typeof createBattlefield>[0] & {
    index: number;
    train?: boolean;
}): Scenario {
    const game = createBattlefield(options);
    const pilots = createPilotsPlugin(game);
    const isTrain = options.train ?? true;

    const fieldSize = options?.size ?? 1000;
    const centerX = fieldSize / 2;
    const centerY = fieldSize / 2;
    const spawnRadius = fieldSize * 0.4; // spawn at 40% from center to edge
    const edgeMargin = 300;

    // Random base diagonal angle (0-360 degrees)
    const baseDiagonalAngle = randomRangeFloat(0, 2 * PI);
    // Max deviation: ±15 degrees (PI/12 radians)
    const maxDeviation = PI / 12;

    const tanks: number[] = [];
    const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

    // Create agent tank (team 0) - with independent deviation from diagonal
    {
        const deviation1 = randomRangeFloat(-maxDeviation, maxDeviation);
        const angle1 = baseDiagonalAngle + deviation1;
        const x = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle1) * spawnRadius));
        const y = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle1) * spawnRadius));
        const playerId = createPlayer(0);
        // Face toward center
        const rotationToCenter = angle1 + PI;
        const tank = createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId,
            teamId: 0,
            x,
            y,
            rotation: rotationToCenter,
            color: [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        tanks.push(tank);
    }

    // Create enemy bot (team 1) - opposite side with its own independent deviation
    {
        const deviation2 = randomRangeFloat(-maxDeviation, maxDeviation);
        const angle2 = baseDiagonalAngle + PI + deviation2; // opposite side + its own deviation
        const x = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerX + Math.cos(angle2) * spawnRadius));
        const y = Math.max(edgeMargin, Math.min(fieldSize - edgeMargin, centerY + Math.sin(angle2) * spawnRadius));
        const playerId = createPlayer(1);
        // Face toward center
        const rotationToCenter = angle2 + PI;
        const tank = createTank({
            type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
            playerId,
            teamId: 1,
            x,
            y,
            rotation: rotationToCenter,
            color: [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
        });
        tanks.push(tank);
    }

    // Place building in the center as objective
    createBuilding({ x: centerX, y: centerY });

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
        move: randomRangeFloat(0, 0.05),
        aim: {
            aimError: randomRangeFloat(0.6, 0.9),
            shootChance: randomRangeFloat(0.01, 0.1),
        },
    }));

    pilots.toggle(true);

    return scenario;
}

