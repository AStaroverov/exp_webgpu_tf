import { PI, sqrt, pow, min } from '../../../../lib/math.ts';
import { randomRangeFloat, randomRangeInt } from '../../../../lib/random.ts';
import { createPlayer } from '../../../tanks/src/Game/ECS/Entities/Player.ts';
import { createTank } from '../../../tanks/src/Game/ECS/Entities/Tank/createTank.ts';
import { VehicleType } from '../../../tanks/src/Game/Config/vehicles.ts';
import { createScenarioCore, ScenarioCoreOptions } from '../createScenarioCore.ts';
import { Scenario } from '../types.ts';
import { BotLevel, createBotFeatures } from './botFeatures.ts';
import { fillWithCurrentAgents } from './fillWithCurrentAgents.ts';
import { fillWithSimpleHeuristicAgents } from './fillWithSimpleHeuristicAgents.ts';

const tankTypes = [VehicleType.LightTank, VehicleType.MediumTank, VehicleType.HeavyTank] as const;

const MIN_RADIUS = 100;

/**
 * Creates a scenario with N agents and M bots at random positions.
 * Follows DRY by providing a generic way to spawn teams.
 */
export function createRandomNvsMScenario(
    options: ScenarioCoreOptions,
    agentsCount: number,
    botsCount: number,
    botLevel: BotLevel = 0,
): Scenario {
    const scenario = createScenarioCore(options);
    const margin = 100;
    const tankPositions: { x: number, y: number }[] = [];

    const getMinDist = (x: number, y: number): number => {
        let minDist = Infinity;
        for (let i = 0; i < tankPositions.length; i++) {
            const tank = tankPositions[i];
            const dist = sqrt(pow(tank.x - x, 2) + pow(tank.y - y, 2));
            minDist = min(minDist, dist);
        }
        return minDist;
    };

    const findSpawnPosition = () => {
        let x: number, y: number, dist = Infinity, j = 0;

        do {
            j++;
            const rx = randomRangeFloat(margin, scenario.width - margin);
            const ry = randomRangeFloat(margin, scenario.height - margin);
            const d = getMinDist(rx, ry);
            if (dist === Infinity || d > dist) {
                dist = d;
                x = rx;
                y = ry;
            }
        } while (dist < MIN_RADIUS * 2 && j < 100);

        return { x: x!, y: y! };
    };

    const createTeamTanks = (count: number, teamId: number) => {
        for (let i = 0; i < count; i++) {
            const { x, y } = findSpawnPosition();
            tankPositions.push({ x, y });

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
        }
    };

    createTeamTanks(agentsCount, 0);
    fillWithCurrentAgents(scenario);

    createTeamTanks(botsCount, 1);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(botLevel));

    return scenario;
}

