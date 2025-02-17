import { DI } from '../../DI';
import { spawnBullet } from '../Components/Bullet.ts';
import { Changed, defineQuery } from 'bitecs';
import { Tank } from '../Components/Tank.ts';
import { resetTankControllerShot, shouldTankControllerShot, TankController } from '../Components/TankController.ts';

export function createSpawnerBulletsSystem({ world } = DI) {
    const queryTanks = defineQuery([Tank, Changed(TankController)]);

    return (() => {
        const tankEids = queryTanks(world);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];

            if (shouldTankControllerShot(tankEid)) {
                spawnBullet(tankEid);
                resetTankControllerShot(tankEid);
            }
        }
    });
}
