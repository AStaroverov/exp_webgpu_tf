import { addComponent } from "bitecs";
import type { World } from "bitecs";
import { defineComponent } from "renderer/src/ECS/utils.ts";
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
export const createExplodableComponent = defineComponent((Explodable, ctx) => {
  const kind = ctx.table.flat(Int8Array);
  const visual = ctx.table.flat(Int8Array);
  const damage = ctx.table.flat(Float64Array);
  const radius = ctx.table.flat(Float64Array);
  const vfxSize = ctx.table.flat(Float64Array);
  const lightRadius = ctx.table.flat(Float64Array);
  return {
    kind,
    visual,
    damage,
    radius,
    vfxSize,
    lightRadius,
    addComponent(world: World, eid: number, settings: ExplodableSettings) {
      addComponent(world, eid, Explodable);
      kind.set(eid, settings.kind);
      visual.set(eid, settings.visual);
      damage.set(eid, settings.damage);
      radius.set(eid, settings.radius);
      vfxSize.set(eid, settings.vfxSize);
      lightRadius.set(eid, settings.lightRadius);
    },
    getDamageKind(eid: number): DamageKind {
      return kind.get(eid) as DamageKind;
    },
    getVisual(eid: number): ExplosionVisual {
      return visual.get(eid) as ExplosionVisual;
    },
  };
});
