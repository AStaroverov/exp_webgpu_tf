import { createScenario, ScenarioConfig } from './createScenario.ts';
import { ScenarioCoreOptions } from './createScenarioCore.ts';
import { Scenario } from './types.ts';
import { SpawnStrategy } from './Utils/spawnStrategies.ts';
import { ObstacleStrategy } from './Utils/obstacleStrategies.ts';
import { AllyPilotStrategy, EnemyPilotStrategy } from './Utils/pilotStrategies.ts';

export type { ScenarioConfig, SpawnStrategy, ObstacleStrategy, AllyPilotStrategy, EnemyPilotStrategy };

// Ordering mirrors the legacy hardcoded `scenarios` array so stored
// `mapScenarioIndexToSuccessRatio` keeps pointing at the same scenarios after the refactor.
export const scenarioCompositions: readonly ScenarioConfig[] = [
    // 0: 1v1 random positions, simplest bot
    {
        teamSize: { allies: 1, enemies: 1 },
        spawn: { kind: 'random' },
        obstacles: { kind: 'single-building' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 0 },
    },
    // 1: 2v2 random positions, simple bot
    {
        teamSize: { allies: 2, enemies: 2 },
        spawn: { kind: 'random' },
        obstacles: { kind: 'single-building' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 0 },
    },
    // 2: 3v3 random positions, simple bot
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'random' },
        obstacles: { kind: 'single-building' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 0 },
    },
    // 3: 2v2 diagonal with central wall obstacle, simple bot
    {
        teamSize: { allies: 2, enemies: 2 },
        spawn: { kind: 'diagonal' },
        obstacles: { kind: 'diagonal-wall' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 0 },
    },
    // 4: 3v3 grid-spawn with fauna, simple bot
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'grid' },
        obstacles: { kind: 'fauna' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 0 },
    },
    // 5: 3v3 random positions, mid bot
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'random' },
        obstacles: { kind: 'single-building' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 1 },
    },
    // 6: 3v3 grid-spawn with fauna, mid bot
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'grid' },
        obstacles: { kind: 'fauna' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'bot', level: 1 },
    },
    // 7: 3v3 grid-spawn with fauna, frozen historical enemies (self-play allies vs historical).
    // Legacy createScenarioWithHistoricalAgents wires allies with CurrentActorAgent via
    // createScenarioGridBase, then fillWithRandomHistoricalAgents fills remaining free
    // vehicles (enemies). Axes are therefore: allyPilot=current, enemyPilot=historical-random.
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'grid' },
        obstacles: { kind: 'fauna' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'historical-random' },
    },
    // 8: 3v3 grid-spawn with fauna, current self-play agents
    {
        teamSize: { allies: 3, enemies: 3 },
        spawn: { kind: 'grid' },
        obstacles: { kind: 'fauna' },
        allyPilot: { kind: 'current' },
        enemyPilot: { kind: 'current' },
    },
] as const;

export function createScenarioFromConfig(config: ScenarioConfig, options: ScenarioCoreOptions): Scenario {
    return createScenario(config, options);
}
