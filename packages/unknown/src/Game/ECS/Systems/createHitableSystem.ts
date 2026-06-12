import { GameDI } from "../../DI/GameDI.ts";
import { scheduleRemoveEntity } from "../Utils/typicalRemoveEntity.ts";
import type { EntityId } from "bitecs";
import { hasComponent, Not, onSet, query } from "bitecs";
import { createChangeDetector } from "../../../../../renderer/src/ECS/Systems/ChangedDetectorSystem.ts";
import { getTankHealth, tearOffTankPart } from "../Entities/Tank/TankUtils.ts";
import { SoundType } from "../Components/Sound.ts";
import { spawnSoundAtParent } from "../Entities/Sound.ts";
import { getGameComponents } from "../createGameWorld.ts";
import { DamageKind } from "../Components/Damagable.ts";
import { EmpStunConfig, FrostSlowConfig } from "../../Config/weapons.ts";
import { findVehicleEidByPartEid } from "../Utils/findPartVehicle.ts";

export function createHitableSystem({ world } = GameDI) {
  const { Hitable, Bullet, VehiclePart, Parent, Vehicle } = getGameComponents(world);
  const hitableChanges = createChangeDetector(world, [onSet(Hitable)]);
  let time = 0;

  return (delta: number) => {
    time += delta;

    if (!hitableChanges.hasChanges()) return;

    const vehiclePartEids = query(world, [VehiclePart, Hitable]);
    const hittedVehicles = new Set<EntityId>();

    for (let i = 0; i < vehiclePartEids.length; i++) {
      const vehiclePartEid = vehiclePartEids[i];
      if (!hitableChanges.has(vehiclePartEid)) continue;

      const vehicleEid = Parent.id.get(Parent.id.get(vehiclePartEid));

      // Readers of the hit ring go first — applyDamage resets it.
      applyKindEffects(vehiclePartEid);
      saveHitters(vehiclePartEid, vehicleEid);
      applyDamage(vehiclePartEid);

      if (hasComponent(world, vehicleEid, Vehicle)) {
        hittedVehicles.add(vehicleEid);
      }

      if (!Hitable.isDestroyed(vehiclePartEid)) continue;

      tearOffTankPart(vehiclePartEid, true);

      getTankHealth(vehicleEid);
    }

    for (const vehicleEid of hittedVehicles) {
      throttledSpawnSoundAtParent(vehicleEid, time, 200);
    }

    const bulletIds = query(world, [Bullet, Hitable]);
    for (let i = 0; i < bulletIds.length; i++) {
      const bulletId = bulletIds[i];
      if (!hitableChanges.has(bulletId)) continue;

      applyDamage(bulletId);

      if (!Hitable.isDestroyed(bulletId)) continue;

      scheduleRemoveEntity(bulletId);
    }

    const restEids = query(world, [Hitable, Not(VehiclePart), Not(Bullet)]);
    for (let i = 0; i < restEids.length; i++) {
      const eid = restEids[i];
      if (!hitableChanges.has(eid)) continue;

      applyDamage(eid);

      if (!Hitable.isDestroyed(eid)) continue;

      scheduleRemoveEntity(eid);
    }

    hitableChanges.clear();
  };
}

// The hit ring carries FINAL damage values (computed by each recorder:
// contact drain, explode, DoT, …) — this is the single place they apply.
function applyDamage(targetEid: number, { world } = GameDI) {
  const { Hitable } = getGameComponents(world);
  const count = Hitable.hitIndex.get(targetEid);

  for (let i = 0; i < count; i++) {
    Hitable.health.set(targetEid, Hitable.health.get(targetEid) - Hitable.getDamage(targetEid, i));
  }
  Hitable.resetHits(targetEid);
}

// Damage-kind specialties, triggered per recorded hit on a vehicle part:
// Frost → one freeze contribution to the part's vehicle (`Slowed` accumulates them);
// Emp → refresh the vehicle's `Stunned` countdown (binary, kind-triggered).
function applyKindEffects(partEid: number, { world } = GameDI) {
  const { Hitable, Slowed, Stunned } = getGameComponents(world);
  const count = Hitable.hitIndex.get(partEid);

  for (let i = 0; i < count; i++) {
    switch (Hitable.getKind(partEid, i)) {
      case DamageKind.Frost: {
        const vehicleEid = findVehicleEidByPartEid(partEid);
        if (vehicleEid === undefined) break; // torn-off debris: explicit absence
        Slowed.addContribution(world, vehicleEid, FrostSlowConfig.freezePerHit);
        break;
      }
      case DamageKind.Emp: {
        const vehicleEid = findVehicleEidByPartEid(partEid);
        if (vehicleEid === undefined) break; // torn-off debris: explicit absence
        Stunned.refresh(vehicleEid, EmpStunConfig.durationMs);
        break;
      }
    }
  }
}

function saveHitters(hittableEid: EntityId, vehicleEid: EntityId, { world } = GameDI) {
  const { Hitable, LastHitters, FriendlyHitters, TeamRef, PlayerRef } = getGameComponents(world);
  if (!hasComponent(world, vehicleEid, LastHitters)) return;
  if (!hasComponent(world, hittableEid, TeamRef)) return;

  const vehiclePartTeamId = TeamRef.id.get(hittableEid);
  const count = Hitable.hitIndex.get(hittableEid);

  for (let i = 0; i < count; i++) {
    const hitEid = Hitable.getSecondEid(hittableEid, i);
    if (!hasComponent(world, hitEid, PlayerRef)) continue;
    if (!hasComponent(world, hitEid, TeamRef)) continue;

    const attackerTeamId = TeamRef.id.get(hitEid);
    const attackerPlayerId = PlayerRef.id.get(hitEid);
    if (attackerTeamId === vehiclePartTeamId) {
      FriendlyHitters.addDamage(vehicleEid, attackerPlayerId, Hitable.getDamage(hittableEid, i));
    } else {
      LastHitters.addDamage(vehicleEid, attackerPlayerId, Hitable.getDamage(hittableEid, i));
    }
  }
}

const mapParentToLastSoundTime = new Map<EntityId, number>();
function throttledSpawnSoundAtParent(parentEid: EntityId, now: number, delay: number) {
  const lastSpawnTime = mapParentToLastSoundTime.get(parentEid);
  if (lastSpawnTime && now - lastSpawnTime < delay) return;
  mapParentToLastSoundTime.set(parentEid, now);

  spawnSoundAtParent({
    parentEid,
    type: SoundType.TankHit,
    loop: false,
    volume: 1,
    autoplay: true,
  });
}
