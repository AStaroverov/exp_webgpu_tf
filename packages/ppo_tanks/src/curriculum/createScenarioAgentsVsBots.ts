import { createScenarioGridBase } from './createScenarioGridBase.ts';
import { Scenario } from './types.ts';
import { BotLevel, createBotFeatures } from './Utils/botFeatures.ts';
import { fillWithSimpleHeuristicAgents } from './Utils/fillWithSimpleHeuristicAgents.ts';

export function createScenarioAgentsVsBots(level: BotLevel, options: Parameters<typeof createScenarioGridBase>[0]): Scenario {
    const scenario = createScenarioGridBase(options);
    fillWithSimpleHeuristicAgents(scenario, createBotFeatures(level));
    return scenario;
}