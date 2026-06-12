import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { BulletSpeedConfig, BulletCaliber, BulletCaliberConfig } from "../../Config/index.ts";

export const MAX_BULLET_SPEED = BulletSpeedConfig.max;
export const MIN_BULLET_SPEED = BulletSpeedConfig.min;

export { BulletCaliber };

export const mapBulletCaliber = BulletCaliberConfig;

export const createBulletComponent = defineComponent((Bullet, ctx) => {
  const caliber = ctx.table.flat(Int8Array);
  return {
    caliber,
    addComponent(world: World, eid: EntityId, calibre: BulletCaliber) {
      addComponent(world, eid, Bullet);
      caliber.set(eid, calibre);
    },
  };
});
