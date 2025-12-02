import { query } from 'bitecs';
import { GameDI } from '../../../DI/GameDI.ts';
import { TankTrack } from '../../Components/TankTrack.ts';
import { Color } from '../../../../../../renderer/src/ECS/Components/Common.ts';

// Initial alpha value for tracks
const INITIAL_ALPHA = 0.4;

export function createUpdateTankTracksSystem({ world } = GameDI) {
    return (delta: number) => {
        const trackEids = query(world, [TankTrack]);

        for (const eid of trackEids) {
            // Update age
            TankTrack.updateAge(eid, delta);

            // Calculate fade based on age progress
            const progress = TankTrack.getProgress(eid);
            const alpha = INITIAL_ALPHA * (1 - progress);

            // Update color alpha
            if (alpha - Color.a[eid] > 0.05) {
                Color.set$(eid, Color.r[eid], Color.g[eid], Color.b[eid], alpha);
            }
        }
    };
}
