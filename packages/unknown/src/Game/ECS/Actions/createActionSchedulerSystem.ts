/**
 * Action scheduler — reaper, nothing else.
 *
 * Once per tick (AFTER the executor systems): while the top action (smallest
 * `seq`) is `Finished`, delete its entity, then re-check the new top. Order is
 * ECS-native (`Action.seq`), so deleting the entity is the whole operation —
 * there is no separate stack to update. It never sets statuses or runs logic.
 */

import { removeEntity } from 'bitecs';
import { ActionScheduleDI, getTopAction } from './ActionScheduleDI.ts';
import { getActionComponents } from './createActionWorld.ts';
import { ActionStatus } from './ActionTypes.ts';

export function createActionSchedulerSystem({ world } = ActionScheduleDI) {
    const { Action } = getActionComponents(world);

    return function scheduler() {
        for (;;) {
            const top = getTopAction();
            if (top === null) return;
            if (Action.status[top] !== ActionStatus.Finished) return;

            removeEntity(world, top);
        }
    };
}
