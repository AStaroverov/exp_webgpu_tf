import { query } from 'bitecs';
import { getRenderWorldComponents } from '../../createRenderWorld.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const INITIAL_ALPHA = 0.4;

export function createUpdateTreadMarksSystem({ renderWorld } = Worlds) {
    const { Color, TreadMark, ProgressFx } = getRenderWorldComponents(renderWorld);

    return () => {
        const treadMarkEids = query(renderWorld, [TreadMark, ProgressFx]);

        for (const eid of treadMarkEids) {
            const progress = ProgressFx.getProgress(eid);
            const alpha = INITIAL_ALPHA * (1 - progress);

            if (alpha - Color.getA(eid) > 0.05) {
                Color.setA$(eid, alpha);
            }
        }
    };
}
