import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { TankTrack } from '../../Components/TankTrack.ts';
import { Progress } from '../../Components/Progress.ts';
import { Color } from '../../../../../../renderer/src/ECS/Components/Common.ts';

// Initial alpha value for tracks
const INITIAL_ALPHA = 0.4;

export function createUpdateTankTracksSystem({ world } = GameDI) {
    return () => {
        const trackEids = query(world, [TankTrack, Progress]);

        for (const eid of trackEids) {
            // Calculate fade based on age progress
            const progress = Progress.getProgress(eid);
            const alpha = INITIAL_ALPHA * (1 - progress);

            // Update color alpha
            if (alpha - Color.a[eid] > 0.05) {
                Color.set$(eid, Color.r[eid], Color.g[eid], Color.b[eid], alpha);
            }
        }
    };
}
