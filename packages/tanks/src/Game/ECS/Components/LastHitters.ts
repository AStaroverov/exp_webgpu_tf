import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

// Stores hit counts per attacker playerId per entity
// Format: [playerId1, count1, playerId2, count2, ...] (pairs)
// 0 playerId means empty slot
const MAX_HITTERS = 5;
const ENTRY_SIZE = 2; // playerId + count

export const HitCounter = component({
    // [playerId, count] x MAX_HITTERS, playerId=0 means empty
    data: NestedArray.f64(MAX_HITTERS * ENTRY_SIZE, delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, HitCounter);
        HitCounter.reset(eid);
    },

    reset(eid: EntityId): void {
        HitCounter.data.getBatch(eid).fill(0);
    },

    addHit(eid: EntityId, playerId: number): void {
        const data = HitCounter.data.getBatch(eid);
        
        // Find existing entry for this playerId
        for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i] === playerId) {
                // Found - increment count
                data[i + 1]++;
                return;
            }
        }
        
        // Not found - find empty slot
        for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i] === 0) {
                data[i] = playerId;
                data[i + 1] = 1;
                return;
            }
        }
        
        // No empty slot - replace the one with lowest count
        let minIndex = 0;
        let minCount = data[1];
        for (let i = ENTRY_SIZE; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i + 1] < minCount) {
                minCount = data[i + 1];
                minIndex = i;
            }
        }
        data[minIndex] = playerId;
        data[minIndex + 1] = 1;
    },

    getHitCount(eid: EntityId, playerId: number): number {
        const data = HitCounter.data.getBatch(eid);
        
        for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i] === playerId) {
                return data[i + 1];
            }
            if (data[i] === 0) break;
        }
        return 0;
    },

    getTotalHits(eid: EntityId): number {
        const data = HitCounter.data.getBatch(eid);
        let total = 0;
        
        for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i] === 0) break;
            total += data[i + 1];
        }
        return total;
    },

    /** Returns iterator of [playerId, hitCount] pairs */
    forEachHitters(eid: EntityId, callback: (playerId: number, hitCount: number) => void): void {
        const data = HitCounter.data.getBatch(eid);
        
        for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
            if (data[i] === 0) break;
            callback(data[i], data[i + 1]);
        }
    },

    /** Returns raw data array (for direct access) */
    getData(eid: EntityId) {
        return HitCounter.data.getBatch(eid);
    },
});

// Keep old name for backwards compatibility
export const LastHitters = HitCounter;
