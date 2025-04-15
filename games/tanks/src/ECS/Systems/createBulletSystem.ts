import { GameDI } from '../../DI/GameDI.ts';
import { Tank } from '../Components/Tank/Tank.ts';
import { TankController } from '../Components/Tank/TankController.ts';
import { query } from 'bitecs';
import { Player } from '../Components/Player.ts';
import { GlobalTransform } from '../../../../../src/ECS/Components/Transform.ts';
import { spawnBullet } from '../Components/Bullet.ts';

export function createSpawnerBulletsSystem({ world } = GameDI) {
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
