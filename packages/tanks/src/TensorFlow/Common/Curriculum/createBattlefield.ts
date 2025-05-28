import { createGame } from '../../../Game/createGame.ts';
import { randomRangeInt } from '../../../../../../lib/random.ts';

export function createBattlefield(options?: { size?: number }) {
    const size = options?.size ?? randomRangeInt(1400, 2000);
    const game = createGame({ width: size, height: size });

    return game;
}

