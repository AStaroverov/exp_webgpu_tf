import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";
import { DamageKind } from "./Damagable.ts";

const HITS_LIMIT = 30;
const HIT_STRIDE = 3; // sourceEid, damage, kind

const ZERO_HITS = new Float64Array(HIT_STRIDE * HITS_LIMIT);

export const createHitableComponent = defineComponent((Hitable, { obs, table }) => {
  const health = table.flat(Float64Array);
  const hitIndex = table.flat(Int8Array);
  const hits = table.nested(Float64Array, HIT_STRIDE * HITS_LIMIT);

  function resetHits(eid: number) {
    hitIndex.set(eid, 0);
    hits.setBatch(eid, ZERO_HITS);
  }

  return {
    health,
    hitIndex,
    hits,

    addComponent(world: World, eid: number, hp: number) {
      addComponent(world, eid, Hitable);
      health.set(eid, hp);
    },
    hit$: obs((eid: number, secondEid: EntityId, damage: number, kind: DamageKind) => {
      if (hitIndex.get(eid) === HITS_LIMIT) {
        console.warn(`[Hitable] Limit on hits`);
        return;
      }
      const index = hitIndex.get(eid) * HIT_STRIDE;
      hits.set(eid, index, secondEid);
      hits.set(eid, index + 1, damage);
      hits.set(eid, index + 2, kind);
      hitIndex.set(eid, hitIndex.get(eid) + 1);
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
      return health.get(eid) <= 0;
    },
  };
});
