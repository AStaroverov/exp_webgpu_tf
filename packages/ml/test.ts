import { frameTasks } from '../../lib/TasksScheduler/frameTasks.ts';
import { prepareInputArrays } from '../ml-common/InputArrays.ts';
import { createGame } from '../tanks/src/Game/createGame.ts';
import { GameDI } from '../tanks/src/Game/DI/GameDI.ts';
import { createPlayer } from '../tanks/src/Game/ECS/Entities/Player.ts';
import { createMediumTank } from '../tanks/src/Game/ECS/Entities/Tank/Medium/MediumTank.ts';
import { createPilotsPlugin } from '../tanks/src/Pilots/createPilotsPlugin.ts';
import { snapshotTankInputTensor } from '../tanks/src/Pilots/Utils/snapshotTankInputTensor.ts';
import { calculateActionReward, WEIGHTS } from './src/Reward/calculateReward.ts';
import { createBuilding } from '../tanks/src/Game/ECS/Entities/Building/Building.ts';
import { createMLPlugin } from '../tanks/src/ML/createMlPlugin.ts';

const game = createGame({ width: 1200, height: 1000 });
const { gameTick, setRenderTarget, enablePlayer, setPlayerVehicle } = game;
const mlPlugin = createMLPlugin(game);
const pilotsPlugin = createPilotsPlugin(game);
pilotsPlugin.toggle(true);

createBuilding({
    x: 500,
    y: 300,
    rotation: 0,
    color: [1, 0, 0, 1],
});

setRenderTarget(document.querySelector('canvas') as HTMLCanvasElement)

const tanks = [
    // createLightTank({
    // createMediumTank({
    createMediumTank({
        playerId: createPlayer(0),
        teamId: 0,
        x: 300,
        y: 300,
        rotation: 0,
        color: [1, 0, 0.5, 1],
    }),
    // createMediumTank({
    //     playerId: createPlayer(1),
    //     teamId: 0,
    //     x: 200,
    //     y: 300,
    //     rotation: 0,
    //     color: [1, 0, 0, 1],
    // }),
    createMediumTank({
        playerId: createPlayer(1),
        teamId: 1,
        x: 800,
        y: 300,
        rotation: Math.PI / 1.3,
        color: [1, 1, 0, 1],
    }),
];

enablePlayer();
setPlayerVehicle(tanks[0]);

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
        const newActionReward = calculateActionReward(tanks[0], WEIGHTS);
        const reward = (actionReward !== undefined ? newActionReward - actionReward : 0);
        
        if (Math.abs(reward) > 0.01) {
            console.log(reward.toFixed(2));
        }
        
        actionReward = newActionReward;

        snapshotTankInputTensor();

        prepareInputArrays(tanks[0], GameDI.width, GameDI.height);
    }

    // const enemyTankPosition = getMatrixTranslation(GlobalTransform.matrix.getBatch(enemyEid));
    // if (enemyTankPosition[1] < 0) {
    //     TankController.setMove$(enemyEid, -1);
    // }
    // if (enemyTankPosition[1] > GameDI.height) {
    //     TankController.setMove$(enemyEid, 1);
    // }
}, 1);