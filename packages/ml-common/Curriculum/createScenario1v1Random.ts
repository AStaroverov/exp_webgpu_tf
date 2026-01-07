import { PI } from '../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../lib/random.ts';
import { createPlayer } from '../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../tanks/src/Game/Config/vehicles.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { add1v1Pilots } from './Utils/add1v1Pilots.ts';
import { Scenario } from './types.ts';

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

/**
 * Simplest scenario: 1 agent vs 1 simple bot at random positions.
 * No fauna, no obstacles - pure 1v1 combat training.
 */
export function createScenario1v1Random(options: ScenarioCoreOptions): Scenario {
    const scenario = createScenarioCore(options);
    const margin = 100;
    const randomPos = () => ({
        x: randomRangeFloat(margin, scenario.width - margin),
        y: randomRangeFloat(margin, scenario.height - margin),
    });

    // Create tanks
    const agentPos = randomPos();
    const agentTankEid = createTank({
        type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
        playerId: createPlayer(0),
        teamId: 0,
        x: agentPos.x,
        y: agentPos.y,
        rotation: PI * randomRangeFloat(0, 2),
        color: [0, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    const botPos = randomPos();
    const botTankEid = createTank({
        type: tankTypes[randomRangeInt(0, tankTypes.length - 1)],
        playerId: createPlayer(1),
        teamId: 1,
        x: botPos.x,
        y: botPos.y,
        rotation: PI * randomRangeFloat(0, 2),
        color: [1, randomRangeFloat(0.2, 0.7), randomRangeFloat(0.2, 0.7), 1],
    });

    // Add pilots
    add1v1Pilots(scenario, agentTankEid, botTankEid);

    return scenario;
}
