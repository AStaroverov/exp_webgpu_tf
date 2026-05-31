/**
 * Wait executor — trivial timer. If the global top action is a Wait, promote it
 * Idle → Running, accumulate elapsed, and Finish when elapsed >= duration.
 */

import { addEntity, World } from 'bitecs';
import { ActionDescriptor, applyTarget } from '../ActionDescriptor.ts';
import { getTopAction } from '../ActionScheduleDI.ts';
import { getActionComponents } from '../createActionWorld.ts';
import { ActionKind, ActionStatus } from '../ActionTypes.ts';

/** Strict params for a Wait action. */
export type WaitParamsSpec = { duration: number };

/** Enqueue spec for a Wait action (no target). */
export type WaitActionSpec = {
    kind: ActionKind.Wait;
    params: WaitParamsSpec;
};

export const WaitActionDescriptor: ActionDescriptor<WaitActionSpec> = {
    kind: ActionKind.Wait,
    createSystem: (actionWorld) => createWaitActionSystem(actionWorld),
    createAction(world, ownerEid, spec, seq) {
        const { Action, WaitParams } = getActionComponents(world);
        const eid = addEntity(world);
        Action.addComponent(world, eid, ActionKind.Wait, ownerEid, seq);
        applyTarget(world, eid);
        WaitParams.addComponent(world, eid, spec.params.duration);
        return eid;
    },
};

export function createWaitActionSystem(world: World) {
    const { Action, WaitParams } = getActionComponents(world);

    return function updateWait(delta: number) {
        const top = getTopAction();
        if (top === null) return;
        if (Action.kind[top] !== ActionKind.Wait) return;
        if (Action.status[top] === ActionStatus.Finished) return;

        if (Action.status[top] === ActionStatus.Idle) {
            Action.setRunning$(top);
        }

        WaitParams.elapsed[top] += delta;

        if (WaitParams.elapsed[top] >= WaitParams.duration[top]) {
            Action.setFinished$(top);
        }
    };
}
