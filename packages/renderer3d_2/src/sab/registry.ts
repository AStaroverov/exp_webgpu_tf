// Shared-buffer registry for the physics-in-worker bridge (plan §3.2).
//
// Both threads must agree, with ZERO negotiation, on which bytes hold which
// component column. They do that by computing byte offsets from a single ordered
// layout table with a PURE function of (stride, banks, capacity) — run the same
// code on both sides and you get byte-identical offsets. Main `allocate()`s the
// SABs; both threads `bindFromSAB()` views into them.
//
// SCOPE — share ONLY what genuinely crosses threads:
//   - RigidBodyState.position / rotation : the pose the worker produces and the
//     renderer consumes. Double-buffered (banks:2) to publish a tear-free pose via
//     one Atomics SEQ flip (plan §5.2).
//   - RigidBodyState.linvel / angvel     : debug-only, single-buffered; tearing
//     tolerated (plan §3.2 table).
// Everything else stays MAIN-PRIVATE on purpose:
//   - Height.value and RigidBodyRef.id are read only by code that runs on MAIN
//     (applyRigidBodyToTransform reads Height; despawn is keyed by eid, not pid),
//     so neither has a real cross-thread reader. The worker backfills pid into its
//     OWN RigidBodyRef.id and keeps a worker-local pid→eid Map. If a genuine
//     cross-thread reader ever appears (e.g. main needs the live pid), promote the
//     column here — until then keeping them private saves SAB memory and avoids a
//     false sharing invariant.
//   - Shape/Color/Rope/LocalTransform/GlobalTransform/... are render-only.

import { delegate } from "../delegate.ts";

// ---- DATA layout (the ordered, stable table) --------------------------------

export type ColumnCtor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Uint32ArrayConstructor
  | Int32ArrayConstructor;

export type ColumnSpec = {
  // Stable string key (NOT JS-object identity): identical on both threads.
  readonly name: string;
  readonly ctor: ColumnCtor;
  readonly stride: number; // floats per entity (e.g. position=3, rotation=4)
  readonly banks: number; // 2 = double-buffered pose; 1 = single-buffered
};

// ORDER IS LOAD-BEARING: offsets are cumulative, so both threads must iterate this
// exact array. Append new bridge columns to the END; never reorder.
export const BRIDGE_COLUMNS: readonly ColumnSpec[] = [
  { name: "RigidBodyState.position", ctor: Float64Array, stride: 3, banks: 2 },
  { name: "RigidBodyState.rotation", ctor: Float64Array, stride: 4, banks: 2 },
  { name: "RigidBodyState.linvel", ctor: Float64Array, stride: 3, banks: 1 },
  { name: "RigidBodyState.angvel", ctor: Float64Array, stride: 3, banks: 1 },
] as const;

// Bumped whenever the table above changes; the worker asserts it matches its own
// computed value so a layout drift fails loud instead of corrupting memory.
// v2: added the OPS SAB (structural-op ring) to the bundle.
export const LAYOUT_VERSION = 2;

export const CAPACITY = delegate.defaultSize; // 30_000; NEVER change without bumping LAYOUT_VERSION

// ---- CONTROL layout (small Int32 slots) -------------------------------------

export const CONTROL = {
  NEXT_EID: 0, // shared monotonic eid counter (plan §4.2)
  SEQ: 1, // pose publish seqcount (plan §5.2)
  PHYS_TIME_MS: 2, // per-publish stamp, kept for later interpolation (plan §7)
} as const;
export const CONTROL_SLOTS = 3;

// ---- OPS layout (structural-op ring, plan §4.3 SAB op-ring) ------------------
//
// A lock-free single-producer (main) / single-consumer (worker) ring. One Int32
// STATE slot per record is the gate: 0 = empty, non-zero = an opcode the worker has
// not consumed yet. Main writes the FLOAT payload then `Atomics.store`s the opcode
// (release); the worker `Atomics.load`s the opcode (acquire), reads the payload,
// executes, then `Atomics.store`s 0 to free the slot ("уводит в 0"). Cursors are
// thread-private; the opcode gate alone carries the cross-thread handoff. The ring
// is also the spawn buffer for ops written before the worker has finished booting —
// they simply sit in the SAB until the worker drains them.
export const OPS_SLOTS = 1024; // plenty for spawn/despawn bursts; ~94 KB total
export const OPS_PAYLOAD_STRIDE = 11; // f64 per record (eid, bodyType, kind, pos×3, dims×3, density, _)
const OPS_STATE_BYTES = OPS_SLOTS * Int32Array.BYTES_PER_ELEMENT;
// Float64 payload must start 8-byte aligned; OPS_SLOTS*4 is already a multiple of 8.
const OPS_PAYLOAD_BYTE_OFFSET = OPS_STATE_BYTES;
const OPS_PAYLOAD_BYTES = OPS_SLOTS * OPS_PAYLOAD_STRIDE * Float64Array.BYTES_PER_ELEMENT;
export const OPS_SAB_BYTES = OPS_PAYLOAD_BYTE_OFFSET + OPS_PAYLOAD_BYTES;

// ---- Pure offset math (identical on both threads) ---------------------------

type ColumnLayout = {
  readonly spec: ColumnSpec;
  readonly byteOffset: number; // start of bank 0
  readonly bankByteStride: number; // bytes between bank 0 and bank 1
  readonly bankFloats: number; // elements in one bank (stride * CAPACITY)
};

export type Layout = {
  readonly columns: ReadonlyMap<string, ColumnLayout>;
  readonly dataByteLength: number;
};

// Pure function of the table + CAPACITY. Each column reserves `banks` contiguous
// bank regions, each `stride * CAPACITY` elements. Banks of one column are
// adjacent so a column's two banks share base+stride addressing.
export function computeLayout(
  columns: readonly ColumnSpec[] = BRIDGE_COLUMNS,
  capacity: number = CAPACITY,
): Layout {
  const map = new Map<string, ColumnLayout>();
  let byteOffset = 0;
  for (const spec of columns) {
    const bankFloats = spec.stride * capacity;
    const bankByteStride = bankFloats * spec.ctor.BYTES_PER_ELEMENT;
    map.set(spec.name, { spec, byteOffset, bankByteStride, bankFloats });
    byteOffset += bankByteStride * spec.banks;
  }
  return { columns: map, dataByteLength: byteOffset };
}

// ---- Allocation (main only) -------------------------------------------------

export type SabBundle = {
  readonly dataSab: SharedArrayBuffer;
  readonly controlSab: SharedArrayBuffer;
  readonly opsSab: SharedArrayBuffer;
  readonly layoutVersion: number;
};

export function allocate(): SabBundle {
  // No fallback (plan §2/§6.4): SAB is the only path. In the browser this needs
  // cross-origin isolation; in node SAB is always available (crossOriginIsolated
  // is undefined there, so don't trip on it).
  if (typeof globalThis.crossOriginIsolated === "boolean" && !globalThis.crossOriginIsolated) {
    throw new Error(
      "sab/registry.allocate: crossOriginIsolated === false — SharedArrayBuffer is " +
        "unavailable. Set COOP/COEP headers (Cross-Origin-Opener-Policy: same-origin, " +
        "Cross-Origin-Embedder-Policy: require-corp). No single-thread fallback exists.",
    );
  }
  const layout = computeLayout();
  const dataSab = new SharedArrayBuffer(layout.dataByteLength);
  const controlSab = new SharedArrayBuffer(CONTROL_SLOTS * Int32Array.BYTES_PER_ELEMENT);
  const opsSab = new SharedArrayBuffer(OPS_SAB_BYTES);
  return { dataSab, controlSab, opsSab, layoutVersion: LAYOUT_VERSION };
}

// Bind the OPS ring views over the received OPS SAB (both threads call identically).
// `state` (Int32, one per slot) is the Atomics gate; `payload` (Float64, strided) holds
// the record fields. The consuming layer (engineSab) keeps the private read/write cursor.
export type OpsRingViews = { state: Int32Array; payload: Float64Array };

export function bindOps(opsSab: ArrayBufferLike): OpsRingViews {
  return {
    state: new Int32Array(opsSab as ArrayBufferLike, 0, OPS_SLOTS),
    payload: new Float64Array(
      opsSab as ArrayBufferLike,
      OPS_PAYLOAD_BYTE_OFFSET,
      OPS_SLOTS * OPS_PAYLOAD_STRIDE,
    ),
  };
}

// ---- Binding (both threads) -------------------------------------------------

// A bound bridge column: pre-computed per-bank byte offsets so the writer/reader
// can address the active bank with no per-call layout math.
export type BoundColumn = {
  readonly name: string;
  readonly ctor: ColumnCtor;
  readonly stride: number;
  readonly banks: number;
  // Byte offset of each bank within the DATA SAB (length === banks).
  readonly bankByteOffsets: readonly number[];
};

export type BoundColumns = ReadonlyMap<string, BoundColumn>;

// Compute the per-bank byte offsets for every bridge column over the given DATA
// SAB. Views are built lazily by the component (via NestedArray's { sab, byteOffset }
// ctor) so this stays a pure address table; both threads call it identically.
export function bindFromSAB(dataSab: ArrayBufferLike, controlSab: ArrayBufferLike, layoutVersion: number): {
  columns: BoundColumns;
  control: Int32Array;
} {
  if (layoutVersion !== LAYOUT_VERSION) {
    throw new Error(
      `sab/registry.bindFromSAB: layout drift — received version ${layoutVersion}, ` +
        `this thread computes ${LAYOUT_VERSION}. Both threads must run identical layout code.`,
    );
  }
  const layout = computeLayout();
  const columns = new Map<string, BoundColumn>();
  for (const [name, col] of layout.columns) {
    const bankByteOffsets: number[] = [];
    for (let b = 0; b < col.spec.banks; b++) {
      bankByteOffsets.push(col.byteOffset + b * col.bankByteStride);
    }
    columns.set(name, {
      name,
      ctor: col.spec.ctor,
      stride: col.spec.stride,
      banks: col.spec.banks,
      bankByteOffsets,
    });
  }
  void dataSab; // address math only; views are built by the consuming component
  return { columns, control: new Int32Array(controlSab as ArrayBufferLike) };
}

// ---- Shared eid counter (plan §4.2) -----------------------------------------

// The ONE and only eid authority for both worlds: an atomic monotonic counter in
// the CONTROL SAB. Never recycled. Mirrors bitecs's "ids start at 1" convention
// (the slot is zero-initialized, so the first nextEid returns 1).
export function nextEid(control: Int32Array): number {
  return Atomics.add(control, CONTROL.NEXT_EID, 1) + 1;
}
