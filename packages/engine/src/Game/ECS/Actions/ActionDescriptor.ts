/**
 * Per-kind action descriptor — one self-contained object per `ActionKind`,
 * co-located with that kind's executor system. Each descriptor knows how to
 * **encode its spec into a queue slot** (write the common buffers + its per-kind
 * params) and how to build its executor system. It is generic over its enqueue
 * spec `S`, so `encode` receives that kind's strictly typed spec directly. The
 * registry collects these objects and derives the whole `EnqueueActionSpec` union
 * from them — adding a kind never touches a switch or a hand-maintained union.
 */

import { ActionKind, ActionTargetSpec, TargetKind } from "./ActionTypes.ts";
import { ActionsQueueComponent } from "../Components/ActionsQueue.ts";

/** Minimal shape every enqueue spec satisfies. */
export type ActionSpecBase = { kind: ActionKind };

export type ActionDescriptor<S extends ActionSpecBase = ActionSpecBase> = {
  kind: S["kind"];
  /** Write this spec into slot `slot` of owner `eid`'s queue (status = Idle). */
  encode: (eid: number, slot: number, spec: S) => void;
  /** Build this kind's executor system (iterates ActionsQueue, runs its-kind fronts). */
  createSystem: () => (delta: number) => void;
};

/** Write a target spec into a queue slot's `targetKind`/`targetVals` (none → None). */
export function encodeTarget(
  q: ActionsQueueComponent,
  eid: number,
  slot: number,
  target?: ActionTargetSpec,
): void {
  if (!target || target.kind === TargetKind.None) {
    q.setTarget(eid, slot, TargetKind.None, 0, 0);
    return;
  }
  q.setTarget(eid, slot, TargetKind.Hex, target.q, target.r);
}
