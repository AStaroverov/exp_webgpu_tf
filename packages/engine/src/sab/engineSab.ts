import {
  bindFromSAB,
  bindOps,
  CONTROL,
  nextEid as nextEidFromControl,
  OPS_SLOTS,
  type BoundColumns,
  type SabBundle,
} from "../../../renderer3d_2/src/sab/registry.ts";
import { decodeOp, type StructuralOp } from "../Physics/opChannel.ts";
import { NestedArray } from "../../../renderer3d_2/src/utils.ts";
import { delegate } from "../../../renderer3d_2/src/delegate.ts";

export type EngineSab = {
  readonly bundle: SabBundle;
  readonly columns: BoundColumns;
  readonly control: Int32Array;
  readonly isProducer: boolean;
  readBank(): number;
  writeBank(): number;
  publish(physTimeMs: number): void;
  nextEid(): number;
  banks(name: string): NestedArray<Float64ArrayConstructor>[];
  pushOp(encode: (payload: Float64Array, slot: number) => number): void;
  drainOps(handler: (op: StructuralOp) => void): void;
};
export type PhysicsRole =
  | { readonly kind: "producer" } // MAIN: allocate fresh SABs
  | { readonly kind: "consumer"; readonly bundle: SabBundle }; // WORKER: bind received bundle

export function bindBundle(bundle: SabBundle, isProducer: boolean): EngineSab {
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

    isProducer,

    // PRODUCER → ring. The caller's `encode(payload, slot)` writes the record's float
    // fields FIRST and returns the opcode; then `Atomics.store` publishes it (release):
    // the consumer only reads the payload after acquiring a non-zero opcode, so it never
    // sees a half-written record. A still-occupied target slot means the consumer is a
    // full ring behind — fail loud rather than silently overwrite an unconsumed op.
    pushOp(encode: (payload: Float64Array, slot: number) => number) {
      const slot = writeCursor % OPS_SLOTS;
      if (Atomics.load(opsState, slot) !== 0) {
        throw new Error(
          `engineSab.pushOp: OPS ring full at slot ${slot} (consumer not draining). ` +
            `Raise OPS_SLOTS or check the worker.`,
        );
      }
      const opcode = encode(opsPayload, slot); // writes payload fields, returns opcode
      Atomics.store(opsState, slot, opcode); // PUBLISH (release)
      writeCursor++;
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
