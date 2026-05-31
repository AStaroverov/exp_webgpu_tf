import { query } from 'bitecs';
import { getRenderWorldComponents } from '../createRenderWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

// Ages fx progress (RenderWorld). All Progress users in Step 1 are render-only fx.
export function createProgressSystem({ renderWorld } = Worlds) {
    const { ProgressFx } = getRenderWorldComponents(renderWorld);

    return (delta: number) => {
        const eids = query(renderWorld, [ProgressFx]);

        for (let i = 0; i < eids.length; i++) {
            ProgressFx.updateAge(eids[i], delta);
        }
    };
}
