import { clamp, isNumber } from 'lodash';
import { lerp } from '../../../../../../lib/math.ts';
import { createGame } from '../../../Game/createGame.ts';

const MAX_SIZE = 2000;
const MIN_SIZE = 1000;

export function createBattlefield(options?: { size?: number, iteration?: number }) {
    const size = options?.size ?? lerp(MIN_SIZE, MAX_SIZE, isNumber(options?.iteration) ? clamp((options.iteration - 50_000) / 100_000, 0, 1) : 1);
    const game = createGame({ width: size, height: size });

    return game;
}

