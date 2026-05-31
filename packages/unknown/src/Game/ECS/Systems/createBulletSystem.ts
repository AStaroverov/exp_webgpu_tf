import { query } from 'bitecs';
import { Worlds } from '../../DI/Worlds.ts';
import { BridgeDI } from '../../DI/BridgeDI.ts';
import { spawnBullet } from '../Entities/Bullet.ts';
import { getPhysicsWorldComponents } from '../createPhysicsWorld.ts';
import { getRenderWorldComponents } from '../createRenderWorld.ts';

export function createSpawnerBulletsSystem({ physicsWorld, renderWorld } = Worlds) {
    const { Firearms, TurretController, VehicleTurret } = getPhysicsWorldComponents(physicsWorld);

    return ((delta: number) => {
        const { Parent } = getRenderWorldComponents(renderWorld);
        const turretEids = query(physicsWorld, [VehicleTurret, TurretController, Firearms]);

        for (let i = 0; i < turretEids.length; i++) {
            const turretEid = turretEids[i];

            Firearms.updateReloading(turretEid, delta);
            if (!TurretController.shouldShoot(turretEid) || Firearms.isReloading(turretEid)) continue;
            Firearms.startReloading(turretEid);

            // turret phys -> turret render -> vehicle render -> vehicle phys
            const vehicleEid = BridgeDI.getPhysicsOf(Parent.id[BridgeDI.getRenderOf(turretEid)]);
            spawnBullet(vehicleEid);
        }
    });
}
