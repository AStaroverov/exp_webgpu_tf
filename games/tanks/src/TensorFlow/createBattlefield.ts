import { createGame } from '../../createGame.ts';
import { createTankRR } from '../ECS/Components/Tank.ts';
import { random } from '../../../../lib/random.ts';
import { floor } from '../../../../lib/math.ts';
import { DI } from '../DI';
import { getDrawState } from './utils.ts';

export function createBattlefield(tanksCount: number) {
    const game = createGame();
    const width = DI.canvas.offsetWidth;
    const height = DI.canvas.offsetHeight;

    for (let i = 0; i < tanksCount; i++) {
        createTankRR({
            x: (i % 3) * width / 3 + width / 3 * random(),
            y: floor(i / 2) * height / 2 + height / 2 * random(),
            rotation: Math.PI / random(),
            color: [random(), random(), random(), 1],
        });
    }

    const gameTick = (delta: number) => {
        game.gameTick(delta, getDrawState());
    };

    return { ...game, gameTick };
}
