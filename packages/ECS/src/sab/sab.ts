import {
  bindFromSAB,
  bindOps,
  CONTROL,
  nextEid as nextEidFromControl,
  OPS_SLOTS,
  type BoundColumns,
  type SabBundle,
} from "./registry.ts";
import type { ComponentSab } from "../component.ts";
import { NestedArray } from "../typedArray.ts";
import { delegate } from "../delegate.ts";

export type Sab = ComponentSab & {
  readonly bundle: SabBundle;
  readonly columns: BoundColumns;
  readonly control: Int32Array;
  publish(physTimeMs: number): void;
  nextEid(): number;
  drainOps(handler: (opcode: number, payload: Float64Array, slot: number) => void): void;
};
export type SabRole =
  | { readonly kind: "producer" } // MAIN: allocate fresh SABs
  | { readonly kind: "consumer"; readonly bundle: SabBundle }; // WORKER: bind received bundle

export function bindBundle(bundle: SabBundle, isProducer: boolean): Sab {
  const { columns, control } = bindFromSAB(bundle.dataSab, bundle.controlSab, bundle.layoutVersion);
  const { state: opsState, payload: opsPayload } = bindOps(bundle.opsSab);

  let writeCursor = 0;
  let readCursor = 0;

  const bankCache = new Map<string, NestedArray<Float64ArrayConstructor>[]>();

  return {
    bundle,
    columns,
    control,
    isProducer,
    nextEid: () => nextEidFromControl(control),
    readBank: () => (Atomics.load(control, CONTROL.SEQ) - 1) & 1,
    writeBank: () => Atomics.load(control, CONTROL.SEQ) & 1,
    publish(physTimeMs: number) {
      Atomics.store(control, CONTROL.PHYS_TIME_MS, physTimeMs | 0);
      Atomics.add(control, CONTROL.SEQ, 1);
      Atomics.notify(control, CONTROL.SEQ);
    },
    banks(name: string) {
      let cached = bankCache.get(name);
      if (!cached) {
        const col = columns.get(name);
        if (!col) throw new Error(`sab.banks: unknown bridge column "${name}"`);
        cached = col.bankByteOffsets.map((byteOffset) =>
          NestedArray.f64(col.stride, delegate.defaultSize, { sab: bundle.dataSab, byteOffset }),
        );
        bankCache.set(name, cached);
      }
      return cached;
    },
    pushOp(encode: (payload: Float64Array, slot: number) => number) {
      const slot = writeCursor % OPS_SLOTS;
      if (Atomics.load(opsState, slot) !== 0) {
        throw new Error(
          `sab.pushOp: OPS ring full at slot ${slot} (consumer not draining). ` +
            `Raise OPS_SLOTS or check the worker.`,
        );
      }
      const opcode = encode(opsPayload, slot); // writes payload fields, returns opcode
      Atomics.store(opsState, slot, opcode); // PUBLISH (release)
      writeCursor++;
    },
    drainOps(handler: (opcode: number, payload: Float64Array, slot: number) => void) {
      for (;;) {
        const slot = readCursor % OPS_SLOTS;
        const opcode = Atomics.load(opsState, slot);
        if (opcode === 0) return; // ring empty up to here
        handler(opcode, opsPayload, slot);
        Atomics.store(opsState, slot, 0); // free the slot for the producer
        readCursor++;
      }
    },
  };
}
