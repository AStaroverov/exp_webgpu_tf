// Engine-side holder for the shared bridge memory (plan §3 / §4.2).
//
// At Step 2 physics still runs INLINE on main — there is no worker yet — but the
// storage seam and the shared eid counter are already the real, worker-ready API:
//   - the bridge pose columns physically live on a DATA SharedArrayBuffer,
//   - eids come from the CONTROL-SAB NEXT_EID atomic counter (never bitecs auto-eid),
//   - a single writer flips the publish bank each step (trivially correct while
//     there is one writer; the worker will own this flip at Step 3).
//
// In node (headless/training) SAB is always available; in the browser allocate()
// asserts crossOriginIsolated. There is no single-thread fallback.

import {
  allocate,
  bindFromSAB,
  bindOps,
  CONTROL,
  nextEid as nextEidFromControl,
  OPS_SLOTS,
  type BoundColumns,
  type SabBundle,
} from "../../../renderer3d_2/src/sab/registry.ts";
import {
  decodeOp,
  encodeOp,
  type StructuralOp,
} from "../Physics/opChannel.ts";
import { NestedArray } from "../../../renderer3d_2/src/utils.ts";
import { delegate } from "../../../renderer3d_2/src/delegate.ts";

export type EngineSab = {
  readonly bundle: SabBundle;
  readonly columns: BoundColumns;
  readonly control: Int32Array;
  // Publish bank for the double-buffered pose. The single writer (inline now, the
  // worker at Step 3) writes the back bank then bumps SEQ to publish it. With one
  // writer the bank is just `SEQ & 1`; we keep an explicit accessor so the active
  // bank is addressed in exactly one place.
  readBank(): number;
  writeBank(): number;
  publish(physTimeMs: number): void;
  // Pull a fresh eid from the shared monotonic counter (the one authority).
  nextEid(): number;
  // Per-bank Float64 views for a registry bridge column, by stable name (length = the
  // column's bank count). Built/cached from the registry layout so a component (via
  // ctx.sab) binds SAB-backed columns by name with no offset math. Satisfies ComponentSab.
  banks(name: string): NestedArray<Float64ArrayConstructor>[];
  // OPS ring (plan §4.3). MAIN pushes structural ops; the WORKER drains them. SPSC:
  // the Int32 state slot is the Atomics gate (0 = empty), cursors are thread-private.
  pushOps(ops: readonly StructuralOp[]): void;
  drainOps(handler: (op: StructuralOp) => void): void;
};

// MAIN: allocate fresh SABs, then bind. The render thread is the eid-allocation
// authority and the source of the bundle handed to the worker via the init message.
export function createEngineSab(): EngineSab {
  return bindBundle(allocate());
}

// WORKER: bind to the SABs RECEIVED from main (plan §3.3). Same memory, same offsets
// — no allocate(); the worker must point at the identical bytes main allocated, so it
// reuses the exact {dataSab, controlSab, layoutVersion} from the init message.
export function createEngineSabFromBundle(bundle: SabBundle): EngineSab {
  return bindBundle(bundle);
}

// Shared post-bind body: allocate vs receive is the ONLY difference between the two
// constructors above; everything below (banks, publish, eid counter) is identical and
// addresses the SAB through the bound views.
function bindBundle(bundle: SabBundle): EngineSab {
  const { columns, control } = bindFromSAB(bundle.dataSab, bundle.controlSab, bundle.layoutVersion);
  const { state: opsState, payload: opsPayload } = bindOps(bundle.opsSab);

  // Thread-private ring cursors (this EngineSab is per-thread): main advances `write`,
  // the worker advances `read`. The shared handoff is the per-slot Atomics opcode gate.
  let writeCursor = 0;
  let readCursor = 0;

  // Cache the per-bank NestedArray views per column name (built once on first use).
  const bankCache = new Map<string, NestedArray<Float64ArrayConstructor>[]>();

  return {
    bundle,
    columns,
    control,
    // last fully-published bank (plan §5.2)
    readBank: () => (Atomics.load(control, CONTROL.SEQ) - 1) & 1,
    // the bank the writer fills before publishing
    writeBank: () => Atomics.load(control, CONTROL.SEQ) & 1,
    publish(physTimeMs: number) {
      Atomics.store(control, CONTROL.PHYS_TIME_MS, physTimeMs | 0);
      Atomics.add(control, CONTROL.SEQ, 1);
      // notify is a no-op for a main-thread poller but harmless and worker-ready.
      Atomics.notify(control, CONTROL.SEQ);
    },
    nextEid: () => nextEidFromControl(control),

    // Per-bank Float64 views for a bridge column, by registry name. Offsets come from the
    // bound layout (both threads compute them identically), so a component never touches
    // byte math — it binds via ctx.sab.banks(name) and picks read/write bank.
    banks(name: string) {
      let cached = bankCache.get(name);
      if (!cached) {
        const col = columns.get(name);
        if (!col) throw new Error(`engineSab.banks: unknown bridge column "${name}"`);
        cached = col.bankByteOffsets.map((byteOffset) =>
          NestedArray.f64(col.stride, delegate.defaultSize, { sab: bundle.dataSab, byteOffset }),
        );
        bankCache.set(name, cached);
      }
      return cached;
    },

    // MAIN → ring. Write the payload FIRST, then `Atomics.store` the opcode (release):
    // the worker only reads the payload after acquiring a non-zero opcode, so it never
    // sees a half-written record. A still-occupied target slot means the worker is a
    // full ring behind — fail loud rather than silently overwrite an unconsumed op.
    pushOps(ops: readonly StructuralOp[]) {
      for (let i = 0; i < ops.length; i++) {
        const slot = writeCursor % OPS_SLOTS;
        if (Atomics.load(opsState, slot) !== 0) {
          throw new Error(
            `engineSab.pushOps: OPS ring full at slot ${slot} (worker not draining). ` +
              `Raise OPS_SLOTS or check the worker.`,
          );
        }
        const opcode = encodeOp(ops[i], opsPayload, slot);
        Atomics.store(opsState, slot, opcode); // PUBLISH (release): payload is written above
        writeCursor++;
      }
    },

    // WORKER ← ring. Drain in cursor order until an empty slot. `Atomics.load` acquires
    // the producer's payload writes; after executing, `Atomics.store(0)` frees the slot
    // ("уводит в 0"). Called at the worker's step phase boundary (never mid-step).
    drainOps(handler: (op: StructuralOp) => void) {
      for (;;) {
        const slot = readCursor % OPS_SLOTS;
        const opcode = Atomics.load(opsState, slot);
        if (opcode === 0) return; // ring empty up to here
        const op = decodeOp(opcode, opsPayload, slot);
        handler(op);
        Atomics.store(opsState, slot, 0); // free the slot for the producer
        readCursor++;
      }
    },
  };
}
