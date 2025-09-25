import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { createGame } from './src/Game/createGame.ts';
import { GameDI } from './src/Game/DI/GameDI.ts';
import { createPlayer } from './src/Game/ECS/Entities/Player.ts';
import { createMediumTank } from './src/Game/ECS/Entities/Tank/Medium/MediumTank.ts';
import { createPilotsPlugin } from './src/Pilots/createPilotsPlugin.ts';
import { snapshotTankInputTensor } from './src/Pilots/Utils/snapshotTankInputTensor.ts';
import { calculateActionReward, calculateStateReward } from './src/TensorFlow/Reward/calculateReward.ts';

const game = createGame({ width: 1200, height: 1000 });
const { gameTick, setRenderTarget } = game;
const pilotsPlugin = createPilotsPlugin(game);
pilotsPlugin.toggle(true);

setRenderTarget(document.querySelector('canvas') as HTMLCanvasElement);

const tanks = [
    // createLightTank({
    // createMediumTank({
    createMediumTank({
        playerId: createPlayer(0),
        teamId: 0,
        x: 300,
        y: 100,
        rotation: 0,
        color: [1, 0, 0.5, 1],
    }),
    createMediumTank({
        playerId: createPlayer(1),
        teamId: 1,
        x: 200,
        y: 300,
        rotation: 0,
        color: [1, 0, 0, 1],
    }),
    createMediumTank({
        playerId: createPlayer(1),
        teamId: 0,
        x: 400,
        y: 300,
        rotation: Math.PI / 1.3,
        color: [1, 1, 0, 1],
    }),
];

pilotsPlugin.setPlayerPilot(tanks[0]);

console.log('>> TANKS ', tanks);

tanks.forEach((tank) => {
    // TankController.setShooting$(tank, 1);
});
// TankController.setShooting$(tanks[1], 1);

// TankController.setMove$(enemyEid, 1);
let i = 0;
let actionReward: undefined | number = undefined;
frameTasks.addInterval(() => {
    i++;

    gameTick(16.66);

    if (i > 10 && i % 3 === 0) {
        const deltaAction = (actionReward ? calculateActionReward(tanks[0]) - actionReward : 0);
        const stateReward = calculateStateReward(tanks[0], GameDI.width, GameDI.height, 1);
        const reward = stateReward + deltaAction;

        snapshotTankInputTensor();
        actionReward = calculateActionReward(tanks[0]);
    }

    // const enemyTankPosition = getMatrixTranslation(GlobalTransform.matrix.getBatch(enemyEid));
    // if (enemyTankPosition[1] < 0) {
    //     TankController.setMove$(enemyEid, -1);
    // }
    // if (enemyTankPosition[1] > GameDI.height) {
    //     TankController.setMove$(enemyEid, 1);
    // }
}, 1);