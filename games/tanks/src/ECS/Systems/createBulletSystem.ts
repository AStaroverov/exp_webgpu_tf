import { DI } from '../../DI';
import { Bullet, spawnBullet } from '../Components/Bullet.ts';
import {
    getMatrixTranslationX,
    getMatrixTranslationY,
    GlobalTransform,
} from '../../../../../src/ECS/Components/Transform.ts';
import { Changed, defineQuery, removeEntity } from 'bitecs';
import { Tank } from '../Components/Tank.ts';
import { resetTankControllerShot, shouldTankControllerShot, TankController } from '../Components/TankController.ts';

export function createSpawnerBulletsSystem({ world, canvas } = DI) {
    const queryTanks = defineQuery([Tank, Changed(TankController)]);
    const queryBullets = defineQuery([Bullet, GlobalTransform]);

    return (() => {
        const tankEids = queryTanks(world);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];

            if (shouldTankControllerShot(tankEid)) {
                spawnBullet(tankEid);
                resetTankControllerShot(tankEid);
            }
        }

        const { width, height } = canvas;
        const bulletsIds = queryBullets(world);

        for (let i = 0; i < bulletsIds.length; i++) {
            const bulletId = bulletsIds[i];
            const bulletGlobalTransform = GlobalTransform.matrix[bulletId];

            const x = getMatrixTranslationX(bulletGlobalTransform);
            const y = getMatrixTranslationY(bulletGlobalTransform);

            if (x < 0 || x > width || y < 0 || y > height) {
                removeEntity(world, bulletId);
            }
        }
    });
}
