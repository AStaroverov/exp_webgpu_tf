import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { Firearms } from '../Components/Firearms.ts';
import { TurretController } from '../Components/TurretController.ts';
import { VehicleTurret } from '../Components/VehicleTurret.ts';
import { Parent } from '../Components/Parent.ts';
import { spawnBullet } from '../Entities/Bullet.ts';

export function createSpawnerBulletsSystem({ world } = GameDI) {
    return ((delta: number) => {
        const turretEids = query(world, [VehicleTurret, TurretController, Firearms]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];

            Firearms.updateReloading(turretEid, delta);
            if (!TurretController.shouldShoot(turretEid) || Firearms.isReloading(turretEid)) continue;
            Firearms.startReloading(turretEid);

            const vehicleEid = Parent.id[turretEid];
            spawnBullet(vehicleEid);
        }
    });
}
