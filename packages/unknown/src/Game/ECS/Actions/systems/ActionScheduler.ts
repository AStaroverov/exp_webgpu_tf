/**
 * Action scheduler — reaper, nothing else (now over the GAME world).
 *
 * Once per tick (AFTER the executor systems): for each `ActionsQueue` whose front
 * (slot 0) is Finished, shift the pending next (slot 1) down into slot 0 across
 * every buffer and decrement `count` (`dropFront`). The new slot 0 is Idle and
 * starts next tick (seamless chain). Slot 1 is left stale but gated out by `count`.
 * It never sets statuses or runs action logic.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { ActionStatus } from '../ActionTypes.ts';

export function createActionSchedulerSystem({ world } = GameDI) {
    const { ActionsQueue } = getGameComponents(world);

    return function scheduler() {
        const eids = query(world, [ActionsQueue]);
        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            if (ActionsQueue.count[eid] > 0 && ActionsQueue.getStatus(eid, 0) === ActionStatus.Finished) {
                ActionsQueue.dropFront(eid);
            }
        }
    };
}
