import { resetSpawnGrid } from '../../../tanks/src/Arena/State/Game/SpawnGrid.ts';
import { createScenarioCore, ScenarioCoreOptions } from './createScenarioCore.ts';
import { Scenario } from './types.ts';
import { applySpawn, SpawnStrategy } from './Utils/spawnStrategies.ts';
import { applyObstacles, ObstacleStrategy } from './Utils/obstacleStrategies.ts';
import {
    AllyPilotStrategy,
    EnemyPilotStrategy,
    applyAllyPilot,
    applyEnemyPilot,
} from './Utils/pilotStrategies.ts';

export type ScenarioConfig = {
    teamSize: { allies: number; enemies: number };
    spawn: SpawnStrategy;
    obstacles: ObstacleStrategy;
    allyPilot: AllyPilotStrategy;
    enemyPilot: EnemyPilotStrategy;
};

export function createScenario(config: ScenarioConfig, options: ScenarioCoreOptions): Scenario {
    // Self-play forces train=true: ally CurrentActorAgents must be created with
    // train=true so their experience buffers feed gradient updates. Mutate options
    // before createScenarioCore so scenario.train reflects the self-play mode for
    // both ally and enemy wiring below.
    const effectiveOptions = config.enemyPilot.kind === 'current'
        ? { ...options, train: true }
        : options;

    const scenario = createScenarioCore(effectiveOptions);
    resetSpawnGrid();

    const tanks = applySpawn(config.spawn, config.teamSize);
    applyObstacles(config.obstacles);
    applyAllyPilot(config.allyPilot, scenario, tanks, config.teamSize.allies);
    applyEnemyPilot(config.enemyPilot, scenario, tanks, config.teamSize);

    return scenario;
}
