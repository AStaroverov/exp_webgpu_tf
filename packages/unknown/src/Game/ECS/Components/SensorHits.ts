import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray, TypedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

const HITS_LIMIT = 10;

/**
 * Per-entity ring of sensor overlaps (mirrors `Hitable.hit$`): the physics
 * callback records which entities a sensor projectile touched this frame;
 * `createApplySensorHitsSystem` consumes the ring. `hitLifeCostMs` is the
 * delivery mechanic: each recorded enemy-part hit eats that much of the
 * projectile's remaining lifetime (pass-through decay).
 */
export const createSensorHitsComponent = defineComponent((SensorHits, obs) => {
  const hitIndex = TypedArray.i8(delegate.defaultSize);
  const hits = NestedArray.f64(HITS_LIMIT, delegate.defaultSize);
  //@todo: separate component for the hitLifeCostMs
  const hitLifeCostMs = TypedArray.f64(delegate.defaultSize);

  function resetHits(eid: number) {
    hitIndex[eid] = 0;
  }

  return {
    hitIndex,
    hits,
    hitLifeCostMs,

    addComponent(world: World, eid: EntityId, hitLifeCost: number) {
      addComponent(world, eid, SensorHits);
      resetHits(eid);
      hitLifeCostMs[eid] = hitLifeCost;
    },
    hit$: obs((eid: number, otherEid: EntityId) => {
      const index = hitIndex[eid];
      if (index === HITS_LIMIT) {
        console.warn(`[SensorHits] Limit on hits`);
        return;
      }
      hits.set(eid, index, otherEid);
      hitIndex[eid] = index + 1;
    }),
    resetHits,
  };
});
