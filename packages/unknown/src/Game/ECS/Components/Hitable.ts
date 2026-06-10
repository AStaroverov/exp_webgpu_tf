import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray, TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { DamageKind } from "./Damagable.ts";

const HITS_LIMIT = 10;
const HIT_STRIDE = 3; // sourceEid, damage, kind

export const createHitableComponent = defineComponent((Hitable, obs) => {
  const health = TypedArray.f64(delegate.defaultSize);
  const hitIndex = TypedArray.i8(delegate.defaultSize);
  const hits = NestedArray.f64(HIT_STRIDE * HITS_LIMIT, delegate.defaultSize);

  function resetHits(eid: number) {
    hitIndex[eid] = 0;
    hits.getBatch(eid).fill(0);
  }

  return {
    health,
    hitIndex,
    hits,

    addComponent(world: World, eid: number, hp: number) {
      addComponent(world, eid, Hitable);
      resetHits(eid);
      health[eid] = hp;
    },
    /**
     * Record a hit: `damage` is the FINAL damage value, fully computed by the
     * caller (contact force × `Damagable`, blast × proximity, DoT tick, …).
     * `createHitableSystem` is the single pipeline that applies it and triggers
     * the `kind` specialty; `secondEid` is kept for attribution (`LastHitters`).
     */
    hit$: obs((eid: number, secondEid: EntityId, damage: number, kind: DamageKind) => {
      if (hitIndex[eid] === HITS_LIMIT) {
        console.warn(`[Hitable] Limit on hits`);
        return;
      }
      const index = hitIndex[eid] * HIT_STRIDE;
      hits.set(eid, index, secondEid);
      hits.set(eid, index + 1, damage);
      hits.set(eid, index + 2, kind);
      hitIndex[eid] = hitIndex[eid] + 1;
    }),
    resetHits,
    getSecondEid(eid: number, hit: number): EntityId {
      return hits.get(eid, hit * HIT_STRIDE);
    },
    getDamage(eid: number, hit: number): number {
      return hits.get(eid, hit * HIT_STRIDE + 1);
    },
    getKind(eid: number, hit: number): DamageKind {
      return hits.get(eid, hit * HIT_STRIDE + 2);
    },
    isDestroyed(eid: number): boolean {
      return health[eid] <= 0;
    },
  };
});
