import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

const HITS_LIMIT = 30;

/**
 * Per-entity ring of sensor overlaps (mirrors `Hitable.hit$`): the physics
 * callback records which entities a sensor projectile touched this frame;
 * `createApplySensorHitsSystem` consumes the ring. `hitLifeCostMs` is the
 * delivery mechanic: each recorded enemy-part hit eats that much of the
 * projectile's remaining lifetime (pass-through decay).
 */
export const createSensorHitsComponent = defineComponent((SensorHits, { obs, table }) => {
  const hits = table.nested(Float64Array, HITS_LIMIT);
  const hitIndex = table.flat(Int8Array);
  //@todo: separate component for the hitLifeCostMs
  const hitLifeCostMs = table.flat(Float64Array);

  function resetHits(eid: number) {
    hitIndex.set(eid, 0);
  }

  return {
    hits,
    hitIndex,
    hitLifeCostMs,

    addComponent(world: World, eid: EntityId, hitLifeCost: number) {
      addComponent(world, eid, SensorHits);
      hitLifeCostMs.set(eid, hitLifeCost);
    },
    hit$: obs((eid: number, otherEid: EntityId) => {
      const index = hitIndex.get(eid);
      if (index === HITS_LIMIT) {
        console.warn(`[SensorHits] Limit on hits`);
        return;
      }
      hits.set(eid, index, otherEid);
      hitIndex.set(eid, index + 1);
    }),
    resetHits,
  };
});
