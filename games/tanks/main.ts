import { createTankRR } from './src/ECS/Components/Tank.ts';
import { createGame } from './createGame.ts';
import { PLAYER_REFS } from './src/consts.ts';

createGame();

// const tankId = createTankRR({
//     x: 500,
//     y: 500,
//     rotation: Math.PI / 1.3,
//     color: [1, 0, 0, 1],
// });

const tankId4 = createTankRR({
    x: 200,
    y: 200,
    rotation: Math.PI / 4,
    color: [1, 0, 1, 1],
});

PLAYER_REFS.tankPid = tankId4;
