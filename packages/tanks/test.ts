import { createGame } from './src/Game/createGame.ts';
import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { GameDI } from './src/Game/DI/GameDI.ts';
import { calculateActionReward, calculateStateReward } from './src/TensorFlow/Reward/calculateReward.ts';
import { createPlayer } from './src/Game/ECS/Entities/Player.ts';
import { snapshotTankInputTensor } from './src/Pilots/Utils/snapshotTankInputTensor.ts';
import { createHeavyTank } from './src/Game/ECS/Entities/Tank/Heavy/HeavyTank.ts';
import { createLightTank } from './src/Game/ECS/Entities/Tank/Light/LightTank.ts';
import { createPilotsPlugin } from './src/Pilots/createPilotsPlugin.ts';

const game = createGame({ width: 1200, height: 1000 });
const { gameTick, setRenderTarget } = game;
const { setPlayerPilot } = createPilotsPlugin(game);

setRenderTarget(document.querySelector('canvas') as HTMLCanvasElement);

const tanks = [
    // createLightTank({
    // createMediumTank({
    createHeavyTank({
        playerId: createPlayer(0),
        teamId: 0,
        x: 200,
        y: 100,
        rotation: 0,
        color: [1, 0, 0.5, 1],
    }),
    // createMediumTank({
    //     playerId: createPlayer(0),
    //     teamId: 0,
    //     x: 700,
    //     y: 500,
    //     rotation: 0,
    //     color: [1, 0, 0, 1],
    // }),
    // createMediumTank({
    //     playerId: createPlayer(0),
    //     teamId: 0,
    //     x: 300,
    //     y: 500,
    //     rotation: Math.PI / 1.3,
    //     color: [1, 0, 0, 1],
    // }),
    // createMediumTank({
    //     playerId: createPlayer(0),
    //     teamId: 0,
    //     x: 150,
    //     y: 700,
    //     rotation: Math.PI / 1.3,
    //     color: [1, 0, 0, 1],
    // }),
    // createLightTank({
    //     playerId: createPlayer(1),
    //     teamId: 1,
    //     x: 200,
    //     y: 900,
    //     rotation: 0,
    //     color: [1, 1, 0, 1],
    // }),
    // createHeavyTank({
    //     playerId: createPlayer(1),
    //     teamId: 1,
    //     x: 700,
    //     y: 300,
    //     rotation: Math.PI / 1.3,
    //     color: [1, 1, 0, 1],
    // }),
    // createMediumTank({
    //     playerId: createPlayer(1),
    //     teamId: 1,
    //     x: 200,
    //     y: 300,
    //     rotation: Math.PI / 1.3,
    //     color: [1, 1, 0, 1],
    // }),
    createLightTank({
        playerId: createPlayer(1),
        teamId: 1,
        x: 400,
        y: 300,
        rotation: Math.PI / 1.3,
        color: [1, 1, 0, 1],
    }),
];

setPlayerPilot(tanks[0]);

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
        const stateReward = calculateStateReward(tanks[0], GameDI.width, GameDI.height);
        const reward = stateReward + deltaAction;

        // console.log('>>', stateReward, deltaAction, reward);

        snapshotTankInputTensor(tanks[0]);
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