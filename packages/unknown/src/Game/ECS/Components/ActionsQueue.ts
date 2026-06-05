import { NestedArray, TypedArray } from '../../../../../renderer/src/utils.ts';
import { delegate } from '../../../../../renderer/src/delegate.ts';
import { addComponent, World } from 'bitecs';
import { defineComponent } from '../../../../../renderer/src/ECS/utils.ts';
import { MAX_QUEUE, PARAMS } from '../Actions/ActionSlot.ts';

/**
 * ActionsQueue — a fixed-size FIFO queue of atomic actions living ON the
 * controlled entity (a game-world component, added to every vehicle in the common
 * tank builder). Slot 0 is the front (the one that runs); FIFO order = slot order
 * (no `seq`). Split-buffer (SoA) layout: each common field gets its own typed,
 * named buffer; only the per-kind params share one generic `NestedArray`.
 *
 * The raw SoA buffers are ENCAPSULATED (closure-local) — callers never touch them
 * nor the slot index math. Access goes through the per-slot accessor methods, all
 * keyed by `(eid, slot)`; `count` (0..MAX_QUEUE) is the only raw field exposed (an
 * idiomatic scalar `count[eid]`), and a slot index >= count is stale.
 */
export const createActionsQueueComponent = defineComponent((ActionsQueue) => {
    const count = TypedArray.u8(delegate.defaultSize);
    const kind = NestedArray.u32(MAX_QUEUE, delegate.defaultSize);
    const status = NestedArray.u32(MAX_QUEUE, delegate.defaultSize);
    const targetVals = NestedArray.f64(MAX_QUEUE * 2, delegate.defaultSize);
    const targetKind = NestedArray.u32(MAX_QUEUE, delegate.defaultSize);
    const requestNext = NestedArray.u32(MAX_QUEUE, delegate.defaultSize);
    const elapsedMs = NestedArray.f64(MAX_QUEUE, delegate.defaultSize);
    const params = NestedArray.f64(MAX_QUEUE * PARAMS, delegate.defaultSize);
    return {
        count,
        addComponent(world: World, eid: number) {
            addComponent(world, eid, ActionsQueue);
            count[eid] = 0;
        },

        getKind(eid: number, slot: number): number {
            return kind.get(eid, slot);
        },
        setKind(eid: number, slot: number, v: number) {
            kind.set(eid, slot, v);
        },

        getStatus(eid: number, slot: number): number {
            return status.get(eid, slot);
        },
        setStatus(eid: number, slot: number, v: number) {
            status.set(eid, slot, v);
        },

        shouldRequestNext(eid: number, slot: number): boolean {
            return requestNext.get(eid, slot) === 1;
        },
        scheduleRequestNext(eid: number, slot: number) {
            requestNext.set(eid, slot, 1);
        },
        clearRequestNext(eid: number, slot: number) {
            requestNext.set(eid, slot, 0);
        },

        getElapsed(eid: number, slot: number): number {
            return elapsedMs.get(eid, slot);
        },
        addElapsed(eid: number, slot: number, delta: number) {
            elapsedMs.set(eid, slot, elapsedMs.get(eid, slot) + delta);
        },
        resetElapsed(eid: number, slot: number) {
            elapsedMs.set(eid, slot, 0);
        },

        getTargetKind(eid: number, slot: number): number {
            return targetKind.get(eid, slot);
        },
        getTargetVal(eid: number, slot: number, i: number): number {
            return targetVals.get(eid, slot * 2 + i);
        },
        setTarget(eid: number, slot: number, kindVal: number, v0: number, v1: number) {
            targetKind.set(eid, slot, kindVal);
            targetVals.set(eid, slot * 2 + 0, v0);
            targetVals.set(eid, slot * 2 + 1, v1);
        },

        getParam(eid: number, slot: number, off: number): number {
            return params.get(eid, slot * PARAMS + off);
        },
        setParam(eid: number, slot: number, off: number, v: number) {
            params.set(eid, slot * PARAMS + off, v);
        },

        /**
         * Shift the queue down by one: copy slot 1 → slot 0 across EVERY buffer
         * (so the pending next becomes the new front) and decrement `count`. Used
         * by the reaper once the front (slot 0) finishes. The source slot is left
         * stale but is gated out by `count`.
         */
        dropFront(eid: number) {
            kind.set(eid, 0, kind.get(eid, 1));
            status.set(eid, 0, status.get(eid, 1));
            requestNext.set(eid, 0, requestNext.get(eid, 1));
            elapsedMs.set(eid, 0, elapsedMs.get(eid, 1));
            targetKind.set(eid, 0, targetKind.get(eid, 1));
            targetVals.set(eid, 0 * 2 + 0, targetVals.get(eid, 1 * 2 + 0));
            targetVals.set(eid, 0 * 2 + 1, targetVals.get(eid, 1 * 2 + 1));
            for (let p = 0; p < PARAMS; p++) {
                params.set(eid, 0 * PARAMS + p, params.get(eid, 1 * PARAMS + p));
            }
            count[eid]--;
        },
    };
});

export type ActionsQueueComponent = ReturnType<typeof createActionsQueueComponent>;
