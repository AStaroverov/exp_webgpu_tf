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
            x: 200 + (i % 3) * width / 3 + 100 * random(),
            y: 200 + floor(i / 3) * height / 3 + 100 * random(),
            rotation: Math.PI / random(),
            color: [random(), random(), random(), 1],
        });
    }

    const gameTick = (delta: number) => {
        game.gameTick(delta, getDrawState());
    };

    return { ...game, gameTick };
}

// let game;
// macroTasks.addInterval(() => {
//     game = createBattlefield(9);
//
//     const stop = macroTasks.addInterval(() => {
//         game.gameTick(32);
//     }, 32);
//
//     macroTasks.addTimeout(() => {
//         stop();
//     }, 900);
//
//     macroTasks.addTimeout(() => {
//         game.destroy();
//     }, 1000);
// }, 1300);
