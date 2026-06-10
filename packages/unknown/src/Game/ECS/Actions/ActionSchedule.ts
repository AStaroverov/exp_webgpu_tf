/**
 * ActionSchedule — public API for the per-owner action queue, now over the GAME
 * world. Actions live ON the controlled entity as the `ActionsQueue` component
 * (slot 0 = front). Thin free functions in the codebase style; the decision-driver
 * seam is unchanged (`needsDecision` + `enqueueAction`).
 */

import { GameDI } from "../../DI/GameDI.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { ActionDescriptor } from "./ActionDescriptor.ts";
import { ActionStatus } from "./ActionTypes.ts";
import { MAX_QUEUE } from "./ActionSlot.ts";
import { ACTION_REGISTRY, EnqueueActionSpec } from "./registry.ts";

export type { EnqueueActionSpec };
export { MAX_QUEUE };

/** Number of live actions queued for `ownerEid` (0..MAX_QUEUE). */
export function queueDepth(ownerEid: number, { world } = GameDI): number {
  const { ActionsQueue } = getGameComponents(world);
  return ActionsQueue.count[ownerEid];
}

/** Whether `ownerEid` has no live actions queued. */
export function isIdle(ownerEid: number, di = GameDI): boolean {
  return queueDepth(ownerEid, di) === 0;
}

/**
 * Whether the decision layer should be asked for this owner's NEXT action: the
 * queue must have room (`count < MAX_QUEUE`) AND the owner must present an open
 * slot — either it is idle (empty queue is an always-open slot) or its running
 * front (slot 0) has raised its request-next flag (§4).
 */
export function needsDecision(ownerEid: number, { world } = GameDI): boolean {
  const { ActionsQueue } = getGameComponents(world);
  const count = ActionsQueue.count[ownerEid];
  if (count >= MAX_QUEUE) return false;
  if (count === 0) return true;
  return ActionsQueue.shouldRequestNext(ownerEid, 0);
}

/**
 * Encode the spec into the owner's next free slot (`count`) via its kind's
 * descriptor, with `status = Idle` and `requestNext = 0`, then bump `count`.
 * Returns `false` if the owner's queue is already full (`count >= MAX_QUEUE`) —
 * the bounded-queue invariant keeps the decision layer from piling up stale
 * decisions; returns `true` on success.
 */
export function enqueueAction(
  ownerEid: number,
  spec: EnqueueActionSpec,
  { world } = GameDI,
): boolean {
  const { ActionsQueue } = getGameComponents(world);
  const slot = ActionsQueue.count[ownerEid];
  if (slot >= MAX_QUEUE) return false;
  // The registry slot for spec.kind is exactly that kind's descriptor, so its
  // encode accepts this spec; widen the lookup for the type system.
  const descriptor = ACTION_REGISTRY[spec.kind] as ActionDescriptor<EnqueueActionSpec>;
  descriptor.encode(ownerEid, slot, spec);
  ActionsQueue.setStatus(ownerEid, slot, ActionStatus.Idle);
  ActionsQueue.clearRequestNext(ownerEid, slot);
  ActionsQueue.resetElapsed(ownerEid, slot);
  ActionsQueue.count[ownerEid] = slot + 1;
  return true;
}
