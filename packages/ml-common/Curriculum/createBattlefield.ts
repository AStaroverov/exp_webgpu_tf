import { createGame } from '../../tanks/src/Game/createGame.ts';
import { createMLPlugin } from '../../tanks/src/ML/createMlPlugin.ts';

const MAX_SIZE = 1600;
const MIN_SIZE = 1000;

export function createBattlefield(options?: { size?: number, iteration?: number }) {
    const size = options?.size ?? 1000;
    // lerp(
    //     MIN_SIZE,
    //     MAX_SIZE,
    //     isNumber(options?.iteration)
    //         ? clamp((options.iteration / LEARNING_STEPS), 0, 1)
    //         : 1
    // );
    const game = createGame({ width: size, height: size });
    const ml = createMLPlugin(game);

    return { ...game, ml };
}

