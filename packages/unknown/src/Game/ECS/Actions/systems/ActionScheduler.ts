/**
 * Action scheduler — watchdog + reaper (now over the GAME world), once per tick
 * AFTER the executor systems.
 *
 * Watchdog: accumulate run time on each owner's front (slot 0); a front older
 * than `MAX_ACTION_MS` is stuck (e.g. a tank physically blocked mid-hop by wreck
 * parts the grid can't see) — zero its movement controller and force it
 * `Finished` so the slot reopens and the decision layer re-decides.
 *
 * Reaper: for each front that is Finished, shift the pending next (slot 1) down
 * into slot 0 across every buffer and decrement `count` (`dropFront`). The new
 * slot 0 is Idle and starts next tick (seamless chain). Slot 1 is left stale but
 * gated out by `count`. It never runs action logic.
 */

import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { getGameComponents } from '../../createGameWorld.ts';
import { ActionStatus } from '../ActionTypes.ts';
import { MAX_ACTION_MS } from '../ActionSlot.ts';

export function createActionSchedulerSystem({ world } = GameDI) {
    const { ActionsQueue, VehicleController } = getGameComponents(world);

    return function scheduler(delta: number) {
        const eids = query(world, [ActionsQueue]);
        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            if (ActionsQueue.count[eid] === 0) continue;

            // Watchdog: time out a stuck front.
            if (ActionsQueue.getStatus(eid, 0) !== ActionStatus.Finished) {
                ActionsQueue.addElapsed(eid, 0, delta);
                if (ActionsQueue.getElapsed(eid, 0) > MAX_ACTION_MS) {
                    VehicleController.setMove$(eid, 0);
                    VehicleController.setRotate$(eid, 0);
                    ActionsQueue.setStatus(eid, 0, ActionStatus.Finished);
                }
            }

            // Reaper: drop a finished front, promote the pending next.
            if (ActionsQueue.getStatus(eid, 0) === ActionStatus.Finished) {
                ActionsQueue.dropFront(eid);
            }
        }
    };
}
