/**
 * setTankWeapon — swap the weapon a tank's turret carries, IN PLACE (no respawn).
 *
 * A turret's armament is one component: `Firearms` (discrete bullet/grenade guns —
 * normal shell, EMP) or `StreamFirearms` (held-fire hoses — flame, frost). The
 * spawner systems stay disjoint by querying for one or the other, so switching the
 * gun is just: drop whichever weapon component is there, add the new one with the
 * caliber from that weapon's vehicle-type config row (the single source of truth).
 * Everything else the guns need (turret, `TurretController`, `SpawnDeltaPosition`)
 * already lives on the turret and is untouched.
 *
 * Intended for the human player's tank only (driven from the weapon-bar UI); the
 * AI/training tanks keep the fixed loadout their factory gives them.
 */

import { hasComponent, removeComponent } from "bitecs";
import { GameDI } from "../../../DI/GameDI.ts";
import { getGameComponents } from "../../createGameWorld.ts";
import { VehicleType } from "../../Components/Vehicle.ts";
import { getTankConfig } from "../../../Config/vehicles.ts";

export type PlayerWeapon = "normal" | "flame" | "frost" | "emp";

/** Each weapon → the vehicle type whose config row carries its caliber. */
const WEAPON_SOURCE: Record<PlayerWeapon, VehicleType> = {
  normal: VehicleType.MediumTank,
  emp: VehicleType.EmpTank,
  flame: VehicleType.FlameTank,
  frost: VehicleType.FrostTank,
};

export function setTankWeapon(tankEid: number, weapon: PlayerWeapon, { world } = GameDI): void {
  const { Tank, Firearms, StreamFirearms, TurretController } = getGameComponents(world);
  const turretEid = Tank.turretEId.get(tankEid);
  if (!turretEid) return;

  // Drop whatever gun is currently mounted and stop any held shot.
  if (hasComponent(world, turretEid, Firearms)) removeComponent(world, turretEid, Firearms);
  if (hasComponent(world, turretEid, StreamFirearms)) {
    removeComponent(world, turretEid, StreamFirearms);
  }
  TurretController.setShooting$(turretEid, 0);

  const config = getTankConfig(WEAPON_SOURCE[weapon]);

  if (weapon === "flame" || weapon === "frost") {
    StreamFirearms.addComponent(world, turretEid, config.stream!.caliber);
    StreamFirearms.emitAccMs.set(turretEid, 0);
    return;
  }

  // normal / emp — a discrete-fire bullet gun. addComponent only sets the caliber;
  // bitecs keeps the eid's old column data, so clear the reload/windup timers.
  Firearms.addComponent(world, turretEid, config.gun!.caliber);
  Firearms.startReloading(turretEid, 0);
  Firearms.startWindup(turretEid, 0);
}
