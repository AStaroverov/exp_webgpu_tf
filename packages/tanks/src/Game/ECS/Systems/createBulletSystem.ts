import { GameDI } from '../../DI/GameDI.ts';
import { Tank } from '../Components/Tank.ts';
import { TankController } from '../Components/TankController.ts';
import { query } from 'bitecs';
import { PlayerRef } from '../Components/PlayerRef.ts';
import { GlobalTransform } from '../../../../../renderer/src/ECS/Components/Transform.ts';

import { spawnBullet } from '../Entities/Bullet.ts';
import { TankTurret } from '../Components/TankTurret.ts';

export function createSpawnerBulletsSystem({ world } = GameDI) {
    return ((delta: number) => {
        const tankEids = query(world, [PlayerRef, GlobalTransform, Tank, TankController]);

        for (let i = 0; i < tankEids.length; i++) {
            const tankEid = tankEids[i];
            const turretEid = Tank.turretEId[tankEid];

            TankTurret.updateReloading(turretEid, delta);
            if (!TankController.shouldShoot(tankEid) || TankTurret.isReloading(turretEid)) continue;
            TankTurret.startReloading(turretEid);

            spawnBullet(tankEid);
        }
    });
}
