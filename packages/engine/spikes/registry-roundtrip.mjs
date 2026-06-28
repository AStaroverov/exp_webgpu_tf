// SPIKE (Step 2) — THROWAWAY. Registry SAB round-trip across node worker_threads.
//
// Extends spike 0c to the REAL registry layout (sab/registry.ts §3.2):
//   - mirror the registry's PURE offset math here (BRIDGE_COLUMNS + computeLayout),
//     then assert the byte offsets are what the production registry computes — this
//     is the "identical on both threads, no negotiation" guarantee;
//   - allocate one DATA SAB (sized to the cumulative layout) + one CONTROL SAB;
//   - main writes a pose into the WRITE bank of RigidBodyState.position+rotation,
//     publishes via Atomics SEQ; a simulated 'worker' binds its OWN views over the
//     SAME SABs and reads back the published pose — byte-identical;
//   - bump NEXT_EID from BOTH threads concurrently and assert it incremented
//     atomically with no lost updates.
//
// No bitecs, no Rapier, no WASM — only the memory + counter seam. Run:
//   node packages/engine/spikes/registry-roundtrip.mjs

import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";

// ---- Mirror of sab/registry.ts (kept in sync; the assertion below pins it) ----
const CAPACITY = 8; // stand-in for delegate.defaultSize (offset math is identical at any N)
const BRIDGE_COLUMNS = [
  { name: "RigidBodyState.position", bytesPer: 8, stride: 3, banks: 2 },
  { name: "RigidBodyState.rotation", bytesPer: 8, stride: 4, banks: 2 },
  { name: "RigidBodyState.linvel", bytesPer: 8, stride: 3, banks: 1 },
  { name: "RigidBodyState.angvel", bytesPer: 8, stride: 3, banks: 1 },
];

// CONTROL Int32 slots (must match registry.CONTROL)
const NEXT_EID = 0;
const SEQ = 1;
const PHYS_TIME_MS = 2;
const CONTROL_SLOTS = 3;

function computeLayout(columns = BRIDGE_COLUMNS, capacity = CAPACITY) {
  const map = new Map();
  let byteOffset = 0;
  for (const spec of columns) {
    const bankFloats = spec.stride * capacity;
    const bankByteStride = bankFloats * spec.bytesPer;
    const bankByteOffsets = [];
    for (let b = 0; b < spec.banks; b++) bankByteOffsets.push(byteOffset + b * bankByteStride);
    map.set(spec.name, { spec, byteOffset, bankByteStride, bankFloats, bankByteOffsets });
    byteOffset += bankByteStride * spec.banks;
  }
  return { columns: map, dataByteLength: byteOffset };
}

// A NestedArray-style f64 view over the SAB at a bank's byte offset (mirrors
// NestedArray.f64(stride, capacity, { sab, byteOffset })).
function bankView(dataSab, col, bank) {
  return new Float64Array(dataSab, col.bankByteOffsets[bank], col.bankFloats);
}

const TEST_EID = 5;
// position(3) + rotation(4): a non-trivial position + a normalized 90°-about-Z quat.
const TEST_POS = [1.5, -2.25, 3.75];
const TEST_ROT = [0, 0, Math.SQRT1_2, Math.SQRT1_2];

if (isMainThread) main();
else workerMain();

function assert(cond, msg) {
  if (!cond) {
    console.error("SPIKE registry-roundtrip: FAIL —", msg);
    process.exit(1);
  }
}

function main() {
  const layout = computeLayout();

  // ---- 0. Pin the layout: offsets must be byte-exact & contiguous ----
  // position: bank0 @0, bank1 @ 3*8*8=192; rotation banks start @ 384, stride 4*8*8=256;
  // linvel @ 384+512=896; angvel @ 896+192=1088; total = 1088+192 = 1280.
  const pos = layout.columns.get("RigidBodyState.position");
  const rot = layout.columns.get("RigidBodyState.rotation");
  const lin = layout.columns.get("RigidBodyState.linvel");
  const ang = layout.columns.get("RigidBodyState.angvel");
  assert(pos.bankByteOffsets[0] === 0, "position bank0 @ 0");
  assert(pos.bankByteOffsets[1] === 3 * CAPACITY * 8, "position bank1 contiguous after bank0");
  assert(rot.bankByteOffsets[0] === 2 * (3 * CAPACITY * 8), "rotation starts after both position banks");
  assert(rot.spec.banks === 2 && lin.spec.banks === 1 && ang.spec.banks === 1, "bank counts: pose=2, vel=1");
  assert(ang.bankByteOffsets[0] + 3 * CAPACITY * 8 === layout.dataByteLength, "dataByteLength is the cumulative end");
  console.log(`MAIN: layout OK — dataByteLength=${layout.dataByteLength} bytes, ${layout.columns.size} columns`);

  // ---- 1. Allocate SABs ----
  const dataSab = new SharedArrayBuffer(layout.dataByteLength);
  const controlSab = new SharedArrayBuffer(CONTROL_SLOTS * Int32Array.BYTES_PER_ELEMENT);
  const control = new Int32Array(controlSab);

  assert(bankView(dataSab, pos, 0).buffer instanceof SharedArrayBuffer, "MAIN: pose view is SAB-backed");
  assert(control.buffer instanceof SharedArrayBuffer, "MAIN: control view is SAB-backed");

  // ---- 2. Allocate a couple eids from the shared counter (nextEid semantics) ----
  // registry.nextEid = Atomics.add(control, NEXT_EID, 1) + 1  → ids start at 1.
  const nextEid = () => Atomics.add(control, NEXT_EID, 1) + 1;
  const e1 = nextEid(); // 1
  const e2 = nextEid(); // 2
  assert(e1 === 1 && e2 === 2, `nextEid monotonic from 1 (got ${e1}, ${e2})`);

  // ---- 3. Write a pose into the WRITE bank, then publish (flip SEQ) ----
  const writeBank = Atomics.load(control, SEQ) & 1; // 0
  const wPos = bankView(dataSab, pos, writeBank);
  const wRot = bankView(dataSab, rot, writeBank);
  for (let i = 0; i < 3; i++) wPos[TEST_EID * 3 + i] = TEST_POS[i];
  for (let i = 0; i < 4; i++) wRot[TEST_EID * 4 + i] = TEST_ROT[i];
  Atomics.store(control, PHYS_TIME_MS, 1234);
  Atomics.add(control, SEQ, 1); // publish

  // ---- 4. Hand SABs to a 'worker', which binds its OWN views and reads back ----
  const worker = new Worker(new URL(import.meta.url), {
    workerData: { dataSab, controlSab, testEid: TEST_EID },
  });

  worker.on("message", (msg) => {
    if (msg.type === "readback") {
      assert(msg.sabBacked, "WORKER: bound views must be SAB-backed");
      for (let i = 0; i < 3; i++) assert(Object.is(msg.pos[i], TEST_POS[i]), `pos[${i}] mismatch`);
      for (let i = 0; i < 4; i++) assert(Object.is(msg.rot[i], TEST_ROT[i]), `rot[${i}] mismatch`);
      console.log(`MAIN: worker read pose back byte-identical (readBank=${msg.readBank}, physTimeMs=${msg.physTimeMs})`);

      // ---- 5. Concurrent NEXT_EID bumps from both threads, no lost updates ----
      const N = 5000;
      worker.postMessage({ type: "bump", n: N });
      let mainGot = 0;
      for (let i = 0; i < N; i++) {
        Atomics.add(control, NEXT_EID, 1);
        mainGot++;
      }
      void mainGot;

      worker.on("message", (m2) => {
        if (m2.type === "bumped") {
          // counter started at 2 (e1,e2), main +N, worker +N  → 2 + 2N.
          const expected = 2 + 2 * N;
          const actual = Atomics.load(control, NEXT_EID);
          assert(actual === expected, `NEXT_EID atomic: expected ${expected}, got ${actual} (lost updates!)`);
          console.log(`MAIN: NEXT_EID after ${2 * N} concurrent atomic bumps = ${actual} (== ${expected}, no lost updates)`);
          console.log("\nSPIKE registry-roundtrip: PASS");
          worker.terminate().then(() => process.exit(0));
        }
      });
    }
  });

  worker.on("error", (err) => {
    console.error("SPIKE registry-roundtrip: FAIL — worker error:", err);
    process.exit(1);
  });
}

function workerMain() {
  const { dataSab, controlSab, testEid } = workerData;
  const layout = computeLayout();
  const control = new Int32Array(controlSab);
  const pos = layout.columns.get("RigidBodyState.position");
  const rot = layout.columns.get("RigidBodyState.rotation");

  // Bind OWN views over the received SABs (registry.bindFromSAB on the worker).
  const seq = Atomics.load(control, SEQ);
  const readBank = (seq - 1) & 1;
  const rPos = bankView(dataSab, pos, readBank);
  const rRot = bankView(dataSab, rot, readBank);
  const sabBacked = rPos.buffer instanceof SharedArrayBuffer && control.buffer instanceof SharedArrayBuffer;

  parentPort.postMessage({
    type: "readback",
    sabBacked,
    readBank,
    physTimeMs: Atomics.load(control, PHYS_TIME_MS),
    pos: Array.from(rPos.subarray(testEid * 3, testEid * 3 + 3)),
    rot: Array.from(rRot.subarray(testEid * 4, testEid * 4 + 4)),
  });

  parentPort.on("message", (msg) => {
    if (msg.type === "bump") {
      for (let i = 0; i < msg.n; i++) Atomics.add(control, NEXT_EID, 1);
      parentPort.postMessage({ type: "bumped" });
    }
  });
}
