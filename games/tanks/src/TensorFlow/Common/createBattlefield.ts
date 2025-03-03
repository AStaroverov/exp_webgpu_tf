import { createGame } from '../../../createGame.ts';
import { createTankRR } from '../../ECS/Components/Tank.ts';
import { random } from '../../../../../lib/random.ts';
import { floor, sqrt } from '../../../../../lib/math.ts';
import { DI } from '../../DI';
import { getDrawState } from './drawState.ts';

export function createBattlefield(tanksCount: number) {
    const rows = Math.ceil(sqrt(tanksCount));
    const cols = Math.ceil(tanksCount / rows);
    const game = createGame();
    const width = DI.canvas.offsetWidth;
    const height = DI.canvas.offsetHeight;
    let tanks = [] as number[];

    for (let i = 0; i < tanksCount; i++) {
        const eid = createTankRR({
            x: (i % rows) * width / rows + width / rows * random(),
            y: floor(i / cols) * height / cols + height / cols * random(),
            rotation: Math.PI / random(),
            color: [random(), random(), random(), 1],
        });

        tanks.push(eid);
    }

    const gameTick = (delta: number) => {
        game.gameTick(delta, getDrawState());
    };

    return { ...game, tanks, gameTick };
}
