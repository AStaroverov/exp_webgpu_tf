import { GameDI } from '../../DI/GameDI.ts';
import { Tank } from '../Components/Tank.ts';
import { TankController } from '../Components/TankController.ts';
import { query } from 'bitecs';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { GlobalTransform } from '../../../../../src/ECS/Components/Transform.ts';

import { spawnBullet } from '../Entities/Bullet.ts';

export function createSpawnerBulletsSystem({ world } = GameDI) {
    return ((delta: number) => {
        const tankEids = query(world, [PlayerRef, GlobalTransform, Tank, TankController]);

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
