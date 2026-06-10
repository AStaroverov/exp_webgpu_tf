import { query, removeComponent } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

/**
 * Expires `Slowed`: contributions are added by the hitable pipeline (Frost-kind
 * damage), this system only counts the shared duration down. The slow itself is
 * read inline at the speed sites (track control, turret rotation).
 */
export function createSlowedExpirySystem({ world } = GameDI) {
    const { Slowed } = getGameComponents(world);

    return (delta: number) => {
        const eids = query(world, [Slowed]);

        // Backwards: removeComponent swap-removes inside the query's dense array.
        for (let i = eids.length - 1; i >= 0; i--) {
            const eid = eids[i];

            Slowed.remaining[eid] -= delta;
            if (Slowed.remaining[eid] <= 0) {
                removeComponent(world, eid, Slowed);
            }
        }
    };
}
