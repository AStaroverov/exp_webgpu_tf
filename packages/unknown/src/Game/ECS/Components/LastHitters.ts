import { addComponent, EntityId, World } from "bitecs";
import { delegate } from "../../../../../renderer/src/delegate.ts";
import { NestedArray } from "../../../../../renderer/src/utils.ts";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

const MAX_HITTERS = 10;
const ENTRY_SIZE = 2;

/**
 * Per-vehicle damage attribution: (attacker playerId, accumulated damage) pairs.
 * Damage — not hit-event count — is the unit, so a 12-damage bullet and 240
 * stream-particle overlaps of 0.05 weigh the same; rewards/kill credit derived
 * from this stay commensurate across weapon types.
 */
export const createLastHittersComponent = defineComponent((LastHitters) => {
  const data = NestedArray.f64(MAX_HITTERS * ENTRY_SIZE, delegate.defaultSize);

  function reset(eid: EntityId) {
    data.getBatch(eid).fill(0);
  }

  return {
    data,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, LastHitters);
      reset(eid);
    },
    reset,
    addDamage(eid: EntityId, playerId: number, damage: number) {
      const arr = data.getBatch(eid);

      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i] === playerId) {
          arr[i + 1] += damage;
          return;
        }
      }

      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i] === 0) {
          arr[i] = playerId;
          arr[i + 1] = damage;
          return;
        }
      }

      let minIndex = 0;
      let minDamage = arr[1];
      for (let i = ENTRY_SIZE; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i + 1] < minDamage) {
          minDamage = arr[i + 1];
          minIndex = i;
        }
      }
      arr[minIndex] = playerId;
      arr[minIndex + 1] = damage;
    },
    getDamage(eid: EntityId, playerId: number): number {
      const arr = data.getBatch(eid);
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i] === playerId) return arr[i + 1];
        if (arr[i] === 0) break;
      }
      return 0;
    },
    getTotalDamage(eid: EntityId): number {
      const arr = data.getBatch(eid);
      let total = 0;
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i] === 0) break;
        total += arr[i + 1];
      }
      return total;
    },
    forEachHitters(eid: EntityId, callback: (playerId: number, damage: number) => void) {
      const arr = data.getBatch(eid);
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (arr[i] === 0) break;
        callback(arr[i], arr[i + 1]);
      }
    },
    getData(eid: EntityId) {
      return data.getBatch(eid);
    },
  };
});
