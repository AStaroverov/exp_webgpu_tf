/**
 * Hold executor — trivial timer. For each owner whose front (slot 0) action is a
 * Hold, promote it Idle → Running, accumulate elapsed (in-slot scratch), and
 * Finish when elapsed >= duration. Owners run concurrently (each has its own front).
 */

import { query } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { ActionDescriptor, encodeTarget } from "../ActionDescriptor.ts";
import { ActionKind, ActionStatus } from "../ActionTypes.ts";
import { HoldParamOffset } from "../ActionSlot.ts";

/**
 * Fixed margin (same time unit as `delta`/`duration`) before the timer elapses at
 * which we open the slot, so the next decision overlaps the tail of the hold. A
 * margin is nicer than on-finish because the policy call hides behind the remaining
 * hold instead of adding a full re-decide gap after completion.
 */
const REQUEST_NEXT_MARGIN = 200;

/** Strict params for a Hold action. */
export type HoldParamsSpec = { duration: number };

/** Enqueue spec for a Hold action (no target). */
export type HoldActionSpec = {
  kind: ActionKind.Hold;
  params: HoldParamsSpec;
};

export const HoldActionDescriptor: ActionDescriptor<HoldActionSpec> = {
  kind: ActionKind.Hold,
  encode(eid, slot, spec) {
    const { ActionsQueue } = getGameComponents(GameDI.world);
    ActionsQueue.setKind(eid, slot, ActionKind.Hold);
    encodeTarget(ActionsQueue, eid, slot);
    ActionsQueue.setParam(eid, slot, HoldParamOffset.duration, spec.params.duration);
    ActionsQueue.setParam(eid, slot, HoldParamOffset.elapsed, 0);
  },
  createSystem: () => createHoldActionSystem(),
};

export function createHoldActionSystem({ world } = GameDI) {
  const { ActionsQueue, Vehicle, RigidBodyState } = getGameComponents(world);

  return function updateHold(delta: number) {
    const eids = query(world, [ActionsQueue, Vehicle, RigidBodyState]);
    for (const ownerEid of eids) {
      if (ActionsQueue.count.get(ownerEid) === 0) continue;
      if (ActionsQueue.getKind(ownerEid, 0) !== ActionKind.Hold) continue;
      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Finished) continue;

      if (ActionsQueue.getStatus(ownerEid, 0) === ActionStatus.Idle) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Running);
      }

      const duration = ActionsQueue.getParam(ownerEid, 0, HoldParamOffset.duration);
      const elapsed = ActionsQueue.getParam(ownerEid, 0, HoldParamOffset.elapsed) + delta;
      ActionsQueue.setParam(ownerEid, 0, HoldParamOffset.elapsed, elapsed);

      // Open the slot a fixed margin before the timer elapses (idempotent set).
      if (elapsed >= duration - REQUEST_NEXT_MARGIN) {
        ActionsQueue.scheduleRequestNext(ownerEid, 0);
      }

      if (elapsed >= duration) {
        ActionsQueue.setStatus(ownerEid, 0, ActionStatus.Finished);
      }
    }
  };
}
