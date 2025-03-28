import { createTankRR } from './src/ECS/Components/Tank.ts';
import { createGame } from './createGame.ts';
import { PLAYER_REFS } from './src/consts.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { calculateReward } from './src/TensorFlow/Common/calculateReward.ts';
import { GameDI } from './src/DI/GameDI.ts';
import { getMatrixTranslation, GlobalTransform } from '../../src/ECS/Components/Transform.ts';
import { TankController } from './src/ECS/Components/TankController.ts';

const { gameTick } = await createGame({ width: 800, height: 800, withRender: true, withPlayer: true });
const tanks = [
    createTankRR({
        x: 200,
        y: 200,
        rotation: Math.PI / 4,
        color: [1, 0, 1, 1],
    }),
    createTankRR({
        x: 500,
        y: 500,
        rotation: 0,
        color: [1, 0, 0, 1],
    }),
    createTankRR({
        x: 200,
        y: 500,
        rotation: Math.PI / 1.3,
        color: [1, 0, 0, 1],
    }),
    //
    // createTankRR({
    //     x: 500,
    //     y: 200,
    //     rotation: Math.PI / 1.3,
    //     color: [1, 0, 0, 1],
    // }),
];
PLAYER_REFS.tankPid = tanks[0];

console.log('>> PLAYER ', PLAYER_REFS.tankPid);
console.log('>> TANKS ', tanks);


const enemyEid = tanks[1];
TankController.setMove$(enemyEid, 1);
let i = 0;
frameTasks.addInterval(() => {
    i++;

    GameDI.shouldCollectTensor = i % 10 === 0;

    gameTick(16.66);

    if (i > 10 && i % 10 === 6) {
        calculateReward(tanks[0], GameDI.width, GameDI.height, 0);
    }

    const enemyTankPosition = getMatrixTranslation(GlobalTransform.matrix.getBatch(enemyEid));
    if (enemyTankPosition[1] < 0) {
        TankController.setMove$(enemyEid, -1);
    }
    if (enemyTankPosition[1] > GameDI.height) {
        TankController.setMove$(enemyEid, 1);
    }
}, 1);