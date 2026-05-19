import { EntityId } from 'bitecs';
import { CurrentActorAgent } from '../../../tanks/src/Plugins/Pilots/Agents/CurrentActorAgent.ts';
import { RandomHistoricalAgent } from '../../../tanks/src/Plugins/Pilots/Agents/RandomHistoricalAgent.ts';
import { Pilot } from '../../../tanks/src/Plugins/Pilots/Components/Pilot.ts';
import { Scenario } from '../types.ts';
import { BotLevel, createBotFeatures } from './botFeatures.ts';
import { fillWithCurrentAgents } from './fillWithCurrentAgents.ts';
import { fillWithRandomHistoricalAgents } from './fillWithRandomHistoricalAgents.ts';
import { fillWithSimpleHeuristicAgents } from './fillWithSimpleHeuristicAgents.ts';

export type AllyPilotStrategy =
    | { kind: 'current' }
    | { kind: 'historical-random' };

export type EnemyPilotStrategy =
    | { kind: 'bot'; level: BotLevel }
    | { kind: 'historical-random' }
    | { kind: 'current' };

export type TeamSize = { allies: number; enemies: number };

export function applyAllyPilot(
    strategy: AllyPilotStrategy,
    scenario: Scenario,
    tanks: readonly EntityId[],
    alliesCount: number,
): void {
    const allyTanks = tanks.slice(0, alliesCount);

    switch (strategy.kind) {
        case 'current': {
            for (const eid of allyTanks) {
                Pilot.addComponent(scenario.world, eid, new CurrentActorAgent(eid, scenario.train));
            }
            return;
        }
        case 'historical-random': {
            for (const eid of allyTanks) {
                Pilot.addComponent(scenario.world, eid, new RandomHistoricalAgent(eid));
            }
            return;
        }
    }
}

export function applyEnemyPilot(
    strategy: EnemyPilotStrategy,
    scenario: Scenario,
    _tanks: readonly EntityId[],
    _teamSize: TeamSize,
): void {
    switch (strategy.kind) {
        case 'bot':
            fillWithSimpleHeuristicAgents(scenario, createBotFeatures(strategy.level));
            return;
        case 'historical-random':
            fillWithRandomHistoricalAgents(scenario);
            return;
        case 'current':
            // Self-play: train=true has already been forced in createScenario before
            // scenarioCore/ally wiring so ally CurrentActorAgents are also trainable.
            fillWithCurrentAgents(scenario);
            return;
    }
}
