import { addComponent, EntityId, World } from 'bitecs';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { NestedArray } from '../../../../../renderer/src/utils.ts';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';

const MAX_HITTERS = 5;
const ENTRY_SIZE = 2;

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
        addHit(eid: EntityId, playerId: number) {
            const arr = data.getBatch(eid);

            for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
                if (arr[i] === playerId) {
                    arr[i + 1]++;
                    return;
                }
            }

            for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
                if (arr[i] === 0) {
                    arr[i] = playerId;
                    arr[i + 1] = 1;
                    return;
                }
            }

            let minIndex = 0;
            let minCount = arr[1];
            for (let i = ENTRY_SIZE; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
                if (arr[i + 1] < minCount) {
                    minCount = arr[i + 1];
                    minIndex = i;
                }
            }
            arr[minIndex] = playerId;
            arr[minIndex + 1] = 1;
        },
        getHitCount(eid: EntityId, playerId: number): number {
            const arr = data.getBatch(eid);
            for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
                if (arr[i] === playerId) return arr[i + 1];
                if (arr[i] === 0) break;
            }
            return 0;
        },
        getTotalHits(eid: EntityId): number {
            const arr = data.getBatch(eid);
            let total = 0;
            for (let i = 0; i < MAX_HITTERS * ENTRY_SIZE; i += ENTRY_SIZE) {
                if (arr[i] === 0) break;
                total += arr[i + 1];
            }
            return total;
        },
        forEachHitters(eid: EntityId, callback: (playerId: number, hitCount: number) => void) {
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
