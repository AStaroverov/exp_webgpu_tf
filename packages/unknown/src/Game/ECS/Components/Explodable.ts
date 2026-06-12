import { addComponent, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { DamageKind } from "./Damagable.ts";
import { ExplosionVisual } from "../../Config/vfx.ts";

export type ExplodableSettings = {
  /** `DamageKind` of the blast's area damage; defaults to `Physical`. */
  kind: DamageKind;
  /** Detonation visuals row key — see `ExplosionVisualConfig`. */
  visual: ExplosionVisual;
  /** Area damage dealt at the epicenter; falls off linearly to zero at `radius`. */
  damage: number;
  /** Damage radius in world pixels. */
  radius: number;
  /** Explosion VFX sprite size; `0` = damage-only blast, no VFX. */
  vfxSize: number;
  /** Light-flash radius; `0` = damage-only blast, no flash. */
  lightRadius: number;
};

/**
 * Marks an entity that detonates when it is destroyed. The explosion is produced
 * uniformly by `createExplodeSystem` whenever the entity also has a `Destroy`
 * component — regardless of what scheduled the destruction (collision, max range,
 * timeout, ...). The settings here drive both the VFX and the area damage.
 */
export const createExplodableComponent = defineComponent((Explodable) => {
  const kind = TypedArray.i8(delegate.defaultSize);
  const visual = TypedArray.i8(delegate.defaultSize);
  const damage = TypedArray.f64(delegate.defaultSize);
  const radius = TypedArray.f64(delegate.defaultSize);
  const vfxSize = TypedArray.f64(delegate.defaultSize);
  const lightRadius = TypedArray.f64(delegate.defaultSize);
  return {
    kind,
    visual,
    damage,
    radius,
    vfxSize,
    lightRadius,
    addComponent(world: World, eid: number, settings: ExplodableSettings) {
      addComponent(world, eid, Explodable);
      kind[eid] = settings.kind;
      visual[eid] = settings.visual;
      damage[eid] = settings.damage;
      radius[eid] = settings.radius;
      vfxSize[eid] = settings.vfxSize;
      lightRadius[eid] = settings.lightRadius;
    },
    getDamageKind(eid: number): DamageKind {
      return kind[eid] as DamageKind;
    },
    getVisual(eid: number): ExplosionVisual {
      return visual[eid] as ExplosionVisual;
    },
  };
});
