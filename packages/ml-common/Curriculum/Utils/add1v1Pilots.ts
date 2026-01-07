import { randomRangeFloat } from '../../../../lib/random.ts';
import { CurrentActorAgent } from '../../../tanks/src/Pilots/Agents/CurrentActorAgent.ts';
import { SimpleBot, SimpleBotFeatures } from '../../../tanks/src/Pilots/Agents/SimpleBot.ts';
import { Pilot } from '../../../tanks/src/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';

const defaultBotFeatures = (): SimpleBotFeatures => ({
    move: randomRangeFloat(0.1, 0.3),
    aim: {
        aimError: randomRangeFloat(0.6, 0.9),
        shootChance: randomRangeFloat(0.01, 0.1),
    },
});

/**
 * Adds agent and bot pilots to a 1v1 scenario.
 */
export function add1v1Pilots(
    scenario: Scenario,
    agentTankEid: number,
    botTankEid: number,
    botFeatures: SimpleBotFeatures = defaultBotFeatures(),
): void {
    Pilot.addComponent(scenario.world, agentTankEid, new CurrentActorAgent(agentTankEid, scenario.train));
    Pilot.addComponent(scenario.world, botTankEid, new SimpleBot(botTankEid, botFeatures));
}

