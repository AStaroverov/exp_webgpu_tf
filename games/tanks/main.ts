import { createGame } from './createGame.ts';
import { PLAYER_REFS } from './src/consts.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { calculateReward } from './src/TensorFlow/Common/calculateReward.ts';
import { GameDI } from './src/DI/GameDI.ts';
import { TenserFlowDI } from './src/DI/TenserFlowDI.ts';
import { createTank } from './src/ECS/Components/Tank/CreateTank.ts';
import { getNewPlayerId } from './src/ECS/Components/Player.ts';

TenserFlowDI.enabled = true;

const { gameTick } = await createGame({ width: 1200, height: 1000, withRender: true, withPlayer: true });
const tanks = [
    createTank({
        playerId: getNewPlayerId(),
        teamId: 0,
        x: 200,
        y: 100,
        rotation: Math.PI / 4,
        color: [1, 0, 0.5, 1],
    }),
    createTank({
        playerId: getNewPlayerId(),
        teamId: 0,
        x: 700,
        y: 500,
        rotation: 0,
        color: [1, 0, 0, 1],
    }),
    createTank({
        playerId: getNewPlayerId(),
        teamId: 0,
        x: 300,
        y: 500,
        rotation: Math.PI / 1.3,
        color: [1, 0, 0, 1],
    }),
    createTank({
        playerId: getNewPlayerId(),
        teamId: 0,
        x: 150,
        y: 700,
        rotation: Math.PI / 1.3,
        color: [1, 0, 0, 1],
    }),
    createTank({
        playerId: getNewPlayerId(),
        teamId: 1,
        x: 200,
        y: 900,
        rotation: 0,
        color: [1, 1, 0, 1],
    }),

    createTank({
        playerId: getNewPlayerId(),
        teamId: 1,
        x: 200,
        y: 300,
        rotation: Math.PI / 1.3,
        color: [1, 1, 0, 1],
    }),
];
PLAYER_REFS.tankPid = tanks[0];

console.log('>> PLAYER ', PLAYER_REFS.tankPid);
console.log('>> TANKS ', tanks);

// TankController.setMove$(enemyEid, 1);
let i = 0;
frameTasks.addInterval(() => {
    i++;

    TenserFlowDI.shouldCollectState = i % 10 === 0;

    gameTick(16.66);

    if (i > 10 && i % 10 === 6) {
        calculateReward(tanks[0], GameDI.width, GameDI.height, 0);
    }

    // const enemyTankPosition = getMatrixTranslation(GlobalTransform.matrix.getBatch(enemyEid));
    // if (enemyTankPosition[1] < 0) {
    //     TankController.setMove$(enemyEid, -1);
    // }
    // if (enemyTankPosition[1] > GameDI.height) {
    //     TankController.setMove$(enemyEid, 1);
    // }
}, 1);