import { query } from 'bitecs';
import { getFxWorldComponents } from '../../createFxWorld.ts';
import { Worlds } from '../../../DI/Worlds.ts';

const INITIAL_ALPHA = 0.4;

export function createUpdateTreadMarksSystem({ fxWorld } = Worlds) {
    const { Color, TreadMark, ProgressFx } = getFxWorldComponents(fxWorld);

    return () => {
        const treadMarkEids = query(fxWorld, [TreadMark, ProgressFx]);

        for (const eid of treadMarkEids) {
            const progress = ProgressFx.getProgress(eid);
            const alpha = INITIAL_ALPHA * (1 - progress);

            if (alpha - Color.getA(eid) > 0.05) {
                Color.setA$(eid, alpha);
            }
        }
    };
}
