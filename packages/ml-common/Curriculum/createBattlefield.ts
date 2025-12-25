import { clamp, isNumber } from 'lodash';
import { lerp } from '../../../lib/math.ts';
import { createGame } from '../../tanks/src/Game/createGame.ts';
import { LEARNING_STEPS } from '../consts.ts';

const MAX_SIZE = 1600;
const MIN_SIZE = 800;

export function createBattlefield(options?: { size?: number, iteration?: number }) {
    const size = options?.size ?? lerp(
        MIN_SIZE,
        MAX_SIZE,
        isNumber(options?.iteration)
            ? clamp((options.iteration / LEARNING_STEPS), 0, 1)
            : 1
    );
    const game = createGame({ width: size, height: size });

    return game;
}

