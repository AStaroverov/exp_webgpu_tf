import { Scenario } from './types.ts';
import { createBattlefield } from './createBattlefield.ts';
import { fillAlliesWithActorAgents } from './Utils/fillAlliesWithActorAgents.ts';
import { createScenarioBase } from './createScenarioBase.ts';

export const indexScenarioWithAlliesActive = 2;

export async function createScenarioWithAlliesActive(options: Parameters<typeof createBattlefield>[0]): Promise<Scenario> {
    const scenario = await createScenarioBase(options);
    scenario.index = indexScenarioWithAlliesActive;

    const destroy = await fillAlliesWithActorAgents(scenario);

    return {
        ...scenario,
        destroy: () => {
            scenario.destroy();
            destroy();
        },
    };
}