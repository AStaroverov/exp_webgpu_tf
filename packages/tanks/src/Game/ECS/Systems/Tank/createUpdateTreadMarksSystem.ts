import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { TreadMark } from '../../Components/TreadMark.ts';
import { Progress } from '../../Components/Progress.ts';
import { Color } from '../../../../../../renderer/src/ECS/Components/Common.ts';

// Initial alpha value for tread marks
const INITIAL_ALPHA = 0.4;

export function createUpdateTreadMarksSystem({ world } = GameDI) {
    return () => {
        const treadMarkEids = query(world, [TreadMark, Progress]);

        for (const eid of treadMarkEids) {
            // Calculate fade based on age progress
            const progress = Progress.getProgress(eid);
            const alpha = INITIAL_ALPHA * (1 - progress);

            // Update color alpha
            if (alpha - Color.getA(eid) > 0.05) {
                Color.setA$(eid, alpha);
            }
        }
    };
}

