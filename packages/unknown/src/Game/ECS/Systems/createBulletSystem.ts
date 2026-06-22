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
      const vehicleEid = Parent.id.get(turretEid);

      Firearms.updateReloading(turretEid, delta);

      if (hasComponent(world, vehicleEid, Vehicle) && hasComponent(world, vehicleEid, Stunned)) {
        continue;
      }

      const stats = BulletCaliberConfig[Firearms.caliber.get(turretEid) as BulletCaliber];

      // Charging a shot: count the windup down; only fire once it elapses.
      if (Firearms.isWindingUp(turretEid)) {
        Firearms.updateWindup(turretEid, delta);
        if (Firearms.isWindingUp(turretEid)) continue;
      } else {
        if (!TurretController.shouldShoot(turretEid) || Firearms.isReloading(turretEid)) {
          continue;
        }
        // Aimed and ready: start the pre-fire windup instead of firing instantly.
        if (stats.delay > 0) {
          Firearms.startWindup(turretEid, stats.delay);
          continue;
        }
      }

      Firearms.startReloading(turretEid, stats.reloadTime);

      spawnBullet(vehicleEid);
    }
  };
}
