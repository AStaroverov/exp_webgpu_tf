import { EntityId, query } from 'bitecs';
import { Vehicle } from '../../tanks/src/Game/ECS/Components/Vehicle.ts';
import { getTeamsCount } from '../../tanks/src/Game/ECS/Components/TeamRef.ts';
import { createBattlefield } from './createBattlefield.ts';
import { Scenario } from './types.ts';
import { getSuccessRatio as computeSuccessRatio, getTeamHealth } from './utils.ts';

export type ScenarioCoreOptions = Parameters<typeof createBattlefield>[0] & {
    index: number;
    train?: boolean;
};

/**
 * Assembles scenario object from game and tanks.
 * This is the common core used by all scenario creators.
 */
export function createScenarioCore(
    options: ScenarioCoreOptions,
): Scenario {
    const game = createBattlefield(options);
    let initialTanks: readonly EntityId[];
    let initialTeamHealth: Record<number, number>;

    const gameTick = game.gameTick;
    game.gameTick = (delta: number) => {
        initialTanks ??= query(game.world, [Vehicle]);
        initialTeamHealth ??= getTeamHealth(initialTanks);
        gameTick(delta);
    };

    return {
        ...game,
        index: options.index,
        train: options.train ?? true,
        getVehicleEids: () => query(game.world, [Vehicle]),
        getTeamsCount,
        getSuccessRatio: () => initialTanks == null || initialTeamHealth == null
            ? 0
            : computeSuccessRatio(0, initialTeamHealth, getTeamHealth(initialTanks)),
    };
}

