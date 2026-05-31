/**
 * Per-kind action descriptor — one self-contained object per `ActionKind`,
 * co-located with that kind's executor system. Each descriptor knows how to
 * **form its own action entity** (create the entity, add `Action`, fill the
 * target, init its params) and how to build its executor system. It is generic
 * over its enqueue spec `S`, so `createAction` receives that kind's strictly
 * typed spec directly. The registry collects these objects and derives the whole
 * `EnqueueActionSpec` union from them — adding a kind never touches a switch or a
 * hand-maintained union.
 */

import { EntityId, World } from 'bitecs';
import { MapWorldId } from '../../Map/HexGrid.ts';
import { ActionKind, ActionTargetSpec, TargetKind } from './ActionTypes.ts';
import { getActionComponents } from './createActionWorld.ts';

/** Minimal shape every enqueue spec satisfies. */
export type ActionSpecBase = { kind: ActionKind };

export type ActionDescriptor<S extends ActionSpecBase = ActionSpecBase> = {
    kind: S['kind'];
    /**
     * Form the full action entity from the spec and return its eid. `seq` is the
     * FIFO order stamp assigned by `enqueueAction` (→ `Action.seq`).
     */
    createAction: (world: World, ownerEid: number, spec: S, seq: number) => EntityId;
    /** Build this kind's executor system. */
    createSystem: () => (delta: number) => void;
};

/** Add the `ActionTarget` component and fill it from a target spec (none → None). */
export function applyTarget(world: World, eid: number, target?: ActionTargetSpec): void {
    const { ActionTarget } = getActionComponents(world);
    ActionTarget.addComponent(world, eid);
    if (!target) return;
    switch (target.kind) {
        case TargetKind.Entity:
            ActionTarget.setEntity$(eid, target.eid, target.worldId ?? MapWorldId.Game);
            break;
        case TargetKind.Hex:
            ActionTarget.setHex$(eid, target.q, target.r);
            break;
        case TargetKind.Point:
            ActionTarget.setPoint$(eid, target.x, target.y);
            break;
        case TargetKind.None:
            break;
    }
}
