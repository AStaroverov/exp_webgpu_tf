/**
 * ActionScheduleDI — DI state for the action schedule, mirroring the
 * GameDI/MapDI singleton pattern. Set/reset in `createGame`. Pure data only:
 * the dedicated action `world` and the monotonic `nextSeq` counter used to
 * stamp FIFO order onto each enqueued action.
 *
 * The "stack" is no longer a JS array — ordering is ECS-native (the `Action.seq`
 * field), so the action world is the single source of truth. The current top is
 * derived from it via `getTopAction` (the live action with the smallest `seq`).
 */

import { EntityId, query } from 'bitecs';
import { getActionComponents } from './createActionWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

export const ActionScheduleDI: {
    /** Monotonic FIFO counter; the next enqueued action gets this seq, then it++. */
    nextSeq: number;
} = {
    nextSeq: 1,
};

/**
 * The current (top) action — the live action entity with the smallest `seq`
 * (FIFO: oldest still-pending action), or `null` if the schedule is empty.
 *
 * Includes a `Finished` top so the scheduler can reap it; executors check the
 * status themselves. Linear scan: action counts are tiny (a per-unit plan), so
 * this is cheaper than maintaining a mirrored array — swap to a heap behind this
 * same signature if it ever grows.
 */
export function getTopAction({ actionWorld } = Worlds): EntityId | null {
    const { Action } = getActionComponents(actionWorld);
    const eids = query(actionWorld, [Action]);
    let best: EntityId | null = null;
    let bestSeq = Infinity;
    for (let i = 0; i < eids.length; i++) {
        const eid = eids[i];
        if (Action.seq[eid] < bestSeq) {
            bestSeq = Action.seq[eid];
            best = eid;
        }
    }
    return best;
}
