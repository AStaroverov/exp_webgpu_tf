import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { BulletCaliber } from "./Bullet.ts";

export const createFirearmsComponent = defineComponent((Firearms, ctx) => {
  const caliber = ctx.table.flat(Int8Array);
  // Remaining reload ms; the duration itself lives in the caliber's global
  // config and is read at the use site, not copied here. The projectile
  // spawn offset is the separate `SpawnDeltaPosition` component.
  const reloading = ctx.table.flat(Float64Array);
  // Remaining pre-fire windup ms; > 0 means the gun is charging the shot. The
  // windup duration lives in the caliber config (`delay`), read at the use site.
  const windup = ctx.table.flat(Float64Array);

  return {
    caliber,
    reloading,
    windup,

    addComponent(world: World, eid: EntityId, cal: BulletCaliber) {
      addComponent(world, eid, Firearms);
      caliber.set(eid, cal);
    },
    isReloading(eid: EntityId): boolean {
      return reloading.get(eid) > 0;
    },
    startReloading(eid: EntityId, durationMs: number) {
      reloading.set(eid, durationMs);
    },
    updateReloading(eid: EntityId, dt: number) {
      reloading.set(eid, reloading.get(eid) - dt);
    },
    isWindingUp(eid: EntityId): boolean {
      return windup.get(eid) > 0;
    },
    startWindup(eid: EntityId, durationMs: number) {
      windup.set(eid, durationMs);
    },
    updateWindup(eid: EntityId, dt: number) {
      windup.set(eid, windup.get(eid) - dt);
    },
    getCaliber(eid: EntityId): BulletCaliber {
      return caliber.get(eid) as BulletCaliber;
    },
  };
});
