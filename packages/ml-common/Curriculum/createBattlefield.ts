import { clamp, isNumber } from 'lodash-es';
import { lerp } from '../../../lib/math.ts';
import { createGame } from '../../tanks/src/Game/createGame.ts';
import { createMLPlugin } from '../../tanks/src/Plugins/ML/createMlPlugin.ts';
import { createPilotsPlugin } from '../../tanks/src/Plugins/Pilots/createPilotsPlugin.ts';
import { LEARNING_STEPS } from '../consts.ts';
import { addEdgeWalls } from './Utils/addEdgeWalls.ts';

const MAX_SIZE = 1400;
const MIN_SIZE = 600;

export function createBattlefield(options?: { size?: number, iteration?: number }) {
    const size = //options?.size ?? 1000;
    lerp(
        MIN_SIZE,
        MAX_SIZE,
        isNumber(options?.iteration)
            ? clamp((options.iteration / LEARNING_STEPS), 0, 1)
            : 1
    );
    const game = createGame({ width: size, height: size });
    addEdgeWalls(size, size);
    const pilots = createPilotsPlugin(game);
    const ml = createMLPlugin(game);

    pilots.toggle(true);
    ml.toggle(true);

    return { ...game, ml, pilots };
}

