import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { getGameComponents } from '../createGameWorld.ts';

export function createProgressSystem({ world } = GameDI) {
    const { Progress } = getGameComponents(world);

    return (delta: number) => {
        const eids = query(world, [Progress]);

        for (let i = 0; i < eids.length; i++) {
            Progress.updateAge(eids[i], delta);
        }
    };
}
