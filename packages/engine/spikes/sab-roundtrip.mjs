// SPIKE 0c — THROWAWAY. SharedArrayBuffer round-trip across node worker_threads.
//
// Proves the §3.1 SAB-view seam + §5.2 sync model in node (the training topology):
//   - one DATA SharedArrayBuffer, two independent Float64Array VIEWS over it (main + worker),
//     laid out exactly like a NestedArray pose column (stride 7 = position(3) + rotation(4));
//   - one CONTROL SharedArrayBuffer (Int32Array) with NEXT_EID / SEQ / PHYS_TIME_MS slots;
//   - worker writes a known pose into the SHARED bytes, then Atomics.store(SEQ)+notify;
//   - main reads via Atomics.load(SEQ) and asserts byte-identical pose;
//   - both sides assert view.buffer instanceof SharedArrayBuffer (no accidental copy);
//   - minimal double-buffer publish: 2 banks, worker writes back bank b = seq & 1,
//     main reads readBank = (seq - 1) & 1.
//
// No WASM, no Rapier, no bitecs — just the memory seam. Run: node spikes/sab-roundtrip.mjs
// (Single file; the worker body is inlined via { eval: true } so the spike stays self-contained.)

import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";

// ---- Shared layout (must be identical on both sides; mirrors registry.ts pure-offset idea) ----
const STRIDE = 7; // position(3) + rotation(4), one f64 pose, like RigidBodyState
const CAPACITY = 8; // a few eids (stand-in for delegate.defaultSize)
const BANKS = 2; // double-buffered pose (§5.2)

// CONTROL Int32 slots
const NEXT_EID = 0; // shared monotonic eid counter (§4.2)
const SEQ = 1; // publish seqcount (§5.2)
const PHYS_TIME_MS = 2; // per-publish stamp (kept for later interpolation)
const CONTROL_SLOTS = 3;

// Per-bank float count + helper to view a given bank as a flat Float64Array.
const BANK_FLOATS = STRIDE * CAPACITY;
const DATA_FLOATS = BANK_FLOATS * BANKS;

function bankView(dataSab, bank) {
  // Each bank is a contiguous [CAPACITY * STRIDE] f64 region inside the one SAB.
  return new Float64Array(dataSab, bank * BANK_FLOATS * Float64Array.BYTES_PER_ELEMENT, BANK_FLOATS);
}

// The known pose the worker will publish for eid 3. Quaternion is a real normalized
// rotation (90° about Z) so a torn read would be obvious, plus a non-trivial position.
const TEST_EID = 3;
const TEST_POSE = [1.5, -2.25, 3.75, 0, 0, Math.SQRT1_2, Math.SQRT1_2];

if (isMainThread) {
  main();
} else {
  workerMain();
}

function main() {
  // ---- 1. Allocate the SABs on main, build views over them ----
  const dataSab = new SharedArrayBuffer(DATA_FLOATS * Float64Array.BYTES_PER_ELEMENT);
  const controlSab = new SharedArrayBuffer(CONTROL_SLOTS * Int32Array.BYTES_PER_ELEMENT);
  const control = new Int32Array(controlSab);

  // Drive eid allocation through the shared counter (§4.2). Burn a few so the worker's
  // adopted eid is meaningful; we hand the worker the eid we want it to write.
  const e0 = Atomics.add(control, NEXT_EID, 1); // 0
  const e1 = Atomics.add(control, NEXT_EID, 1); // 1
  void e0;
  void e1;

  // Both sides must see SAB-backed memory (the #1 correctness assert in the plan).
  assert(bankView(dataSab, 0).buffer instanceof SharedArrayBuffer, "MAIN: data view is NOT SAB-backed");
  assert(control.buffer instanceof SharedArrayBuffer, "MAIN: control view is NOT SAB-backed");

  const log = [];
  log.push(`MAIN: crossOriginIsolated is browser-only; in node SAB needs no isolation flag.`);
  log.push(`MAIN: data view buffer instanceof SharedArrayBuffer = true`);

  // ---- 2. Spawn worker, post the SABs (structured clone shares, does NOT copy, the SAB) ----
  const worker = new Worker(new URL(import.meta.url), {
    workerData: { dataSab, controlSab, testEid: TEST_EID, testPose: TEST_POSE },
  });

  worker.on("message", (msg) => {
    if (msg.type === "done") {
      // ---- 3. Main reads via the seqcount, then reads the published bank ----
      const seq = Atomics.load(control, SEQ);
      assert(seq >= 1, `MAIN: expected SEQ>=1 after worker publish, got ${seq}`);
      const readBank = (seq - 1) & 1; // last fully-published bank (§5.2)
      const view = bankView(dataSab, readBank);

      const base = TEST_EID * STRIDE;
      const got = Array.from(view.subarray(base, base + STRIDE));

      log.push(`MAIN: read SEQ=${seq} -> readBank=${readBank}, physTimeMs=${Atomics.load(control, PHYS_TIME_MS)}`);
      log.push(`MAIN: worker reported it wrote into back bank=${msg.writtenBank}`);
      log.push(`MAIN: expected pose = [${TEST_POSE.join(", ")}]`);
      log.push(`MAIN: got pose      = [${got.join(", ")}]`);

      // ---- byte-identical assertion (bit-for-bit, not approx) ----
      assert(readBank === msg.writtenBank, `MAIN: readBank ${readBank} != worker writtenBank ${msg.writtenBank}`);
      for (let i = 0; i < STRIDE; i++) {
        assert(
          Object.is(got[i], TEST_POSE[i]),
          `MAIN: pose[${i}] mismatch: got ${got[i]} expected ${TEST_POSE[i]}`,
        );
      }

      // The OTHER bank must still be all-zero — proves the worker wrote exactly one bank
      // and main read the correct one (publish/read flip works, no cross-bank bleed).
      const otherBank = readBank ^ 1;
      const other = bankView(dataSab, otherBank);
      let otherSum = 0;
      for (let i = 0; i < other.length; i++) otherSum += other[i];
      assert(otherSum === 0, `MAIN: non-read bank ${otherBank} unexpectedly non-zero (sum=${otherSum})`);
      log.push(`MAIN: non-read bank ${otherBank} is all-zero (no cross-bank bleed) — double-buffer flip OK`);

      // eid is shared & monotonic: worker adopted TEST_EID, counter never re-handed it out.
      assert(
        Atomics.load(control, NEXT_EID) === 2,
        `MAIN: NEXT_EID drifted; expected 2, got ${Atomics.load(control, NEXT_EID)}`,
      );

      log.push("");
      log.push("SPIKE 0c: PASS — SAB pose round-trip is byte-identical across worker_threads.");
      console.log(log.join("\n"));

      worker.terminate().then(() => process.exit(0));
    }
  });

  worker.on("error", (err) => {
    console.error("SPIKE 0c: FAIL — worker error:", err);
    process.exit(1);
  });
}

function workerMain() {
  const { dataSab, controlSab, testEid, testPose } = workerData;
  const control = new Int32Array(controlSab);

  // Worker binds its OWN views over the SAME SABs (mimics registry.bindFromSAB on the
  // received buffers). These must be SAB-backed too — if structured clone had copied,
  // this would be a plain ArrayBuffer and the round-trip would silently fail.
  assert(new Float64Array(dataSab).buffer instanceof SharedArrayBuffer, "WORKER: data view is NOT SAB-backed");
  assert(control.buffer instanceof SharedArrayBuffer, "WORKER: control view is NOT SAB-backed");

  // §5.2 publish: write the just-"stepped" pose into BACK bank b = seq & 1, then bump SEQ.
  const seq = Atomics.load(control, SEQ); // 0 on first publish
  const backBank = seq & 1;
  const view = bankView(dataSab, backBank);

  const base = testEid * STRIDE;
  for (let i = 0; i < STRIDE; i++) view[base + i] = testPose[i];

  // Stamp physics time, then publish: store(SEQ, ++seq) + notify (no-op for a poller, harmless).
  Atomics.store(control, PHYS_TIME_MS, 42);
  Atomics.store(control, SEQ, seq + 1);
  Atomics.notify(control, SEQ);

  parentPort.postMessage({ type: "done", writtenBank: backBank });
}

function assert(cond, msg) {
  if (!cond) {
    console.error("SPIKE 0c: FAIL —", msg);
    process.exit(1);
  }
}
