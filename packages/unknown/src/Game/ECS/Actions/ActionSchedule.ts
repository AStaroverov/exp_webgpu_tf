/**
 * ActionSchedule — public API for the global action stack.
 *
 * Thin free functions in the codebase style. Action entities live in the
 * dedicated action world (`ActionScheduleDI.world`); the `ownerEid` they carry
 * references a game-world entity (disjoint id space).
 */

import { EntityId, query, removeEntity } from 'bitecs';
import { ActionDescriptor } from './ActionDescriptor.ts';
import { ActionScheduleDI } from './ActionScheduleDI.ts';
import { getActionComponents } from './createActionWorld.ts';
import { ACTION_REGISTRY, EnqueueActionSpec } from './registry.ts';

export type { EnqueueActionSpec };

/**
 * Form the action entity via its kind's descriptor (status = Idle) and push it
 * onto the global stack. Returns the action eid.
 */
export function enqueueAction(
    ownerEid: number,
    spec: EnqueueActionSpec,
    di = ActionScheduleDI,
): EntityId {
    // The registry slot for spec.kind is exactly that kind's descriptor, so its
    // createAction accepts this spec; widen the lookup for the type system.
    const descriptor = ACTION_REGISTRY[spec.kind] as ActionDescriptor<EnqueueActionSpec>;
    // Stamp the next FIFO order onto the action; ordering is ECS-native (Action.seq).
    return descriptor.createAction(di.world, ownerEid, spec, di.nextSeq++);
}

/**
 * Cancel an action: just delete its entity. Order is ECS-native, so there is no
 * separate stack to clean up — the top simply recomputes from the world.
 */
export function cancelAction(actionEid: EntityId, { world } = ActionScheduleDI): void {
    removeEntity(world, actionEid);
}

/** Delete all actions whose ownerEid matches. */
export function clearForOwner(ownerEid: number, { world } = ActionScheduleDI): void {
    const { Action } = getActionComponents(world);
    const eids = query(world, [Action]);
    for (let i = 0; i < eids.length; i++) {
        const eid = eids[i];
        if (Action.ownerEid[eid] === ownerEid) removeEntity(world, eid);
    }
}
