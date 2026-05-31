import { TypedArray } from '../../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../../renderer/src/ECS/utils.ts';
import { ActionKind, ActionStatus } from '../ActionTypes.ts';

/**
 * Action — core status + identity of an action entity. Ordering is ECS-native:
 * the `seq` field holds the FIFO sequence number (assigned at enqueue). The
 * current "top" action is just the live entity with the smallest `seq`
 * (`getTopAction`), so the world is the single source of truth — there is no
 * separate stack to keep in sync.
 */
export const createActionComponent = defineComponent((Action, obs) => {
    const kind = new Uint16Array(delegate.defaultSize);
    const status = TypedArray.u8(delegate.defaultSize);
    const ownerEid = TypedArray.f64(delegate.defaultSize);
    const seq = TypedArray.f64(delegate.defaultSize);
    return {
        kind,
        status,
        ownerEid,
        seq,
        addComponent(world: World, eid: number, k: ActionKind, owner: number, order: number) {
            addComponent(world, eid, Action);
            status[eid] = ActionStatus.Idle;
            kind[eid] = k;
            ownerEid[eid] = owner;
            seq[eid] = order;
        },
        setStatus$: obs((eid: number, s: ActionStatus) => {
            status[eid] = s;
        }),
        setRunning$: obs((eid: number) => {
            status[eid] = ActionStatus.Running;
        }),
        setFinished$: obs((eid: number) => {
            status[eid] = ActionStatus.Finished;
        }),
    };
});
