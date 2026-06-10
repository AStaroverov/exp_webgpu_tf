import { query } from 'bitecs';
import { GameDI } from '../../DI/GameDI.ts';
import { spawnBullet } from '../Entities/Bullet.ts';
import { getGameComponents } from '../createGameWorld.ts';
import { BulletCaliber } from '../Components/Bullet.ts';
import { BulletCaliberConfig } from '../../Config/weapons.ts';

export function createSpawnerBulletsSystem({ world } = GameDI) {
    const { Firearms, TurretController, VehicleTurret, Parent } = getGameComponents(world);

    return ((delta: number) => {
        const turretEids = query(world, [VehicleTurret, TurretController, Firearms]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];

            Firearms.updateReloading(turretEid, delta);
            if (!TurretController.shouldShoot(turretEid) || Firearms.isReloading(turretEid)) continue;
            Firearms.startReloading(turretEid, BulletCaliberConfig[Firearms.caliber[turretEid] as BulletCaliber].reloadTime);

            const vehicleEid = Parent.id[turretEid];
            spawnBullet(vehicleEid);
        }
    });
}
