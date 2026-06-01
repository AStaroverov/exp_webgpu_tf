import { query, removeEntity } from 'bitecs';
import { getFxWorldComponents } from '../createFxWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

// fx-only timeout destroy (FxWorld). fx entities have no Rapier body, so they are
// reaped directly from FxWorld (no Bridge/atom involved).
export function createDestroyByTimeoutSystem({ fxWorld } = Worlds) {
    const { DestroyByTimeoutFx } = getFxWorldComponents(fxWorld);

    return (delta: number) => {
        const eids = query(fxWorld, [DestroyByTimeoutFx]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];

            DestroyByTimeoutFx.updateTimeout(eid, delta);

            if (DestroyByTimeoutFx.timeout[eid] <= 0) {
                removeEntity(fxWorld, eid);
            }
        }
    };
}
