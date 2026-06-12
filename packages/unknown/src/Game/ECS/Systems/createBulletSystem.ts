import { hasComponent, query } from "bitecs";
import { GameDI } from "../../DI/GameDI.ts";
import { spawnBullet } from "../Entities/Bullet.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { BulletCaliber } from "../Components/Bullet.ts";
import { BulletCaliberConfig } from "../../Config/weapons.ts";

export function createSpawnerBulletsSystem({ world } = GameDI) {
  const { Firearms, TurretController, VehicleTurret, Parent, Stunned, Vehicle } =
    getGameComponents(world);

  return (delta: number) => {
    const turretEids = query(world, [VehicleTurret, TurretController, Firearms]);

    for (let i = 0; i < turretEids.length; i++) {
      const turretEid = turretEids[i];
      const vehicleEid = Parent.id[turretEid];

      Firearms.updateReloading(turretEid, delta);

      if (hasComponent(world, vehicleEid, Vehicle) && hasComponent(world, vehicleEid, Stunned)) {
        continue;
      }
      if (!TurretController.shouldShoot(turretEid) || Firearms.isReloading(turretEid)) {
        continue;
      }

      Firearms.startReloading(
        turretEid,
        BulletCaliberConfig[Firearms.caliber[turretEid] as BulletCaliber].reloadTime,
      );

      spawnBullet(vehicleEid);
    }
  };
}
