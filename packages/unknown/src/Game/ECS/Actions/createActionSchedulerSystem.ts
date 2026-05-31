/**
 * Action scheduler — reaper, nothing else.
 *
 * Once per tick (AFTER the executor systems): while the top action (smallest
 * `seq`) is `Finished`, delete its entity, then re-check the new top. Order is
 * ECS-native (`Action.seq`), so deleting the entity is the whole operation —
 * there is no separate stack to update. It never sets statuses or runs logic.
 */

import { removeEntity } from 'bitecs';
import { getTopAction } from './ActionScheduleDI.ts';
import { Worlds } from '../../DI/Worlds.ts';
import { getActionComponents } from './createActionWorld.ts';
import { ActionStatus } from './ActionTypes.ts';

export function createActionSchedulerSystem({ actionWorld } = Worlds) {
    const { Action } = getActionComponents(actionWorld);

    return function scheduler() {
        for (;;) {
            const top = getTopAction();
            if (top === null) return;
            if (Action.status[top] !== ActionStatus.Finished) return;

            removeEntity(actionWorld, top);
        }
    };
}
