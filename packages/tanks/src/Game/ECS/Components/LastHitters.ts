import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray } from '../../../../../renderer/src/utils.ts';
import { component } from '../../../../../renderer/src/ECS/utils.ts';

// Stores last 3 attacker playerIds per entity
// 0 means empty slot
const HITTERS_LIMIT = 3;

export const LastHitters = component({
    // [playerId] x 3, 0 = empty
    hitters: NestedArray.f64(HITTERS_LIMIT, delegate.defaultSize),

    addComponent(world: World, eid: EntityId): void {
        addComponent(world, eid, LastHitters);
        LastHitters.reset(eid);
    },

    reset(eid: EntityId): void {
        LastHitters.hitters.getBatch(eid).fill(0);
    },

    addHitter(eid: EntityId, playerId: number): void {
        const hitters = LastHitters.hitters.getBatch(eid);
        
        // Check if playerId already exists in the list
        let existingIndex = -1;
        for (let i = 0; i < HITTERS_LIMIT && hitters[i] !== 0; i++) {
            if (hitters[i] === playerId) {
                existingIndex = i;
                break;
            }
        }
        
        if (existingIndex === 0) {
            // Already at front, nothing to do
            return;
        }
        
        // Find how many entries we have
        let count = 0;
        for (let i = 0; i < HITTERS_LIMIT && hitters[i] !== 0; i++) {
            count++;
        }
        
        // Shift entries to make room at front
        // If existing, shift only up to that index; otherwise shift all
        const shiftEnd = existingIndex > 0 ? existingIndex : Math.min(count, HITTERS_LIMIT - 1);
        for (let i = shiftEnd; i > 0; i--) {
            hitters[i] = hitters[i - 1];
        }
        
        // Add/move playerId to front
        hitters[0] = playerId;
    },

    getAllHitters(eid: EntityId) {
        return LastHitters.hitters.getBatch(eid)
    },
});
