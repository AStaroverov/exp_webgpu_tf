import { query, removeEntity } from 'bitecs';
import { getRenderWorldComponents } from '../createRenderWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

// fx-only timeout destroy (RenderWorld). fx entities have no Rapier body, so they are
// reaped directly from RenderWorld (no Bridge/atom involved).
export function createDestroyByTimeoutSystem({ renderWorld } = Worlds) {
    const { DestroyByTimeoutFx } = getRenderWorldComponents(renderWorld);

    return (delta: number) => {
        const eids = query(renderWorld, [DestroyByTimeoutFx]);

        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];

            DestroyByTimeoutFx.updateTimeout(eid, delta);

            if (DestroyByTimeoutFx.timeout[eid] <= 0) {
                removeEntity(renderWorld, eid);
            }
        }
    };
}
