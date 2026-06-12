import { addComponent } from "bitecs";
import type { EntityId, World } from "bitecs";
import { defineComponent } from "../../../../../renderer/src/ECS/utils.ts";

const MAX_HITTERS = 10;
const ENTRY_SIZE = 2;

/**
 * Per-vehicle damage attribution: (attacker playerId, accumulated damage) pairs.
 * Damage — not hit-event count — is the unit, so a 12-damage bullet and 240
 * stream-particle overlaps of 0.05 weigh the same; rewards/kill credit derived
 * from this stay commensurate across weapon types.
 */
export const createLastHittersComponent = defineComponent((LastHitters, ctx) => {
  const data = ctx.table.nested(Float64Array, MAX_HITTERS * ENTRY_SIZE);
  const zeroes = new Float64Array(MAX_HITTERS * ENTRY_SIZE);
  const scratch = new Float64Array(MAX_HITTERS * ENTRY_SIZE);

  function reset(eid: EntityId) {
    data.setBatch(eid, zeroes);
  }

  return {
    data,
    addComponent(world: World, eid: EntityId) {
      addComponent(world, eid, LastHitters);
    },
    reset,
    addDamage(eid: EntityId, playerId: number, damage: number) {
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i) === playerId) {
          data.set(eid, i + 1, data.get(eid, i + 1) + damage);
          return;
        }
      }

      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i) === 0) {
          data.set(eid, i, playerId);
          data.set(eid, i + 1, damage);
          return;
        }
      }

      let minIndex = 0;
      let minDamage = data.get(eid, 1);
      for (let i = ENTRY_SIZE; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i + 1) < minDamage) {
          minDamage = data.get(eid, i + 1);
          minIndex = i;
        }
      }
      data.set(eid, minIndex, playerId);
      data.set(eid, minIndex + 1, damage);
    },
    getDamage(eid: EntityId, playerId: number): number {
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i) === playerId) return data.get(eid, i + 1);
        if (data.get(eid, i) === 0) break;
      }
      return 0;
    },
    getTotalDamage(eid: EntityId): number {
      let total = 0;
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i) === 0) break;
        total += data.get(eid, i + 1);
      }
      return total;
    },
    forEachHitters(eid: EntityId, callback: (playerId: number, damage: number) => void) {
      for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
        if (data.get(eid, i) === 0) break;
        callback(data.get(eid, i), data.get(eid, i + 1));
      }
    },
    getData(eid: EntityId) {
      return data.getBatch(eid, scratch);
    },
  };
});
