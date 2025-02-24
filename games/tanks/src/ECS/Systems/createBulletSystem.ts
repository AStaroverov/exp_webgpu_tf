import { DI } from '../../DI';
import { spawnBullet } from '../Components/Bullet.ts';
import { Tank } from '../Components/Tank.ts';
import { TankController } from '../Components/TankController.ts';
import { query } from 'bitecs';
import { Player } from '../Components/Player.ts';
import { GlobalTransform } from '../../../../../src/ECS/Components/Transform.ts';

export function createSpawnerBulletsSystem({ world } = DI) {
    return ((delta: number) => {
        const tankEids = query(world, [Player, GlobalTransform, Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];

            TankController.updateCooldown(tankEid, delta);

            if (TankController.shouldShoot(tankEid)) {
                spawnBullet(tankEid);
                TankController.startCooldown(tankEid);
            }
        }
    });
}
