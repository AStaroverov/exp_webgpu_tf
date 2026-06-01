import { query } from 'bitecs';
import { getFxWorldComponents } from '../createFxWorld.ts';
import { Worlds } from '../../DI/Worlds.ts';

// Ages fx progress (FxWorld). All Progress users are fx entities.
export function createProgressSystem({ fxWorld } = Worlds) {
    const { ProgressFx } = getFxWorldComponents(fxWorld);

    return (delta: number) => {
        const eids = query(fxWorld, [ProgressFx]);

        for (let i = 0; i < eids.length; i++) {
            ProgressFx.updateAge(eids[i], delta);
        }
    };
}
