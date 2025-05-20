import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { createScenarioWithAlliesStatic } from './createScenarioWithAlliesStatic.ts';

export const indexScenarioSoloStatic = 0;

export async function createScenarioSoloStatic(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioWithAlliesStatic(options);
    scenario.index = indexScenarioSoloStatic;
    return scenario;
}