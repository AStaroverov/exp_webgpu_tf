import { addComponent, EntityId, World } from "bitecs";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { BulletSpeedConfig, BulletCaliber, BulletCaliberConfig } from "../../Config/index.ts";

export const MAX_BULLET_SPEED = BulletSpeedConfig.max;
export const MIN_BULLET_SPEED = BulletSpeedConfig.min;

export { BulletCaliber };

export const mapBulletCaliber = BulletCaliberConfig;

export const createBulletComponent = defineComponent((Bullet) => {
  const caliber = TypedArray.i8(delegate.defaultSize);
  return {
    caliber,
    addComponent(world: World, eid: EntityId, calibre: BulletCaliber) {
      addComponent(world, eid, Bullet);
      caliber[eid] = calibre;
    },
  };
});
