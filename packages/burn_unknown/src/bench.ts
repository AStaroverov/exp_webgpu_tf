/**
 * bench — head-to-head v4 inference benchmark, TensorFlow.js vs Burn (wasm),
 * BOTH on the WebGPU backend, in the same browser tab. Hardcoded throwaway page
 * (bench.html). One question: per-decision inference latency, batch = 1, with
 * BOTH sides doing IDENTICAL work.
 *
 * The fair unit of work = exactly what Burn's `V4Trainer.act` does:
 *   forward (policy logits + value) → softmax over the masked logits → read
 *   probs + value back to CPU → sample one action on the CPU.
 *
 * So the tfjs side here is NOT a bare `predict`: it also applies the action mask,
 * softmaxes, reads the probs+value back, and samples — the same pipeline — so the
 * comparison is apples-to-apples. Batch = 1 only (no batching on either side).
 *
 * Every timed iteration awaits the GPU readback (real kernel execution, not just
 * dispatch). A warmup phase absorbs backend init, autotune, kernel compilation.
 */

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

import {
  createPolicyNetwork,
  createValueNetwork,
} from "../../ppo_unknown/src/models/createUnknownNetworks.ts";
import {
  BOARD_ROWS,
  BOARD_COLS,
  BOARD_CHANNELS,
  BOARD_CELLS,
} from "../../ppo_unknown/src/models/dims.ts";
import { ACTION_DIM_TOTAL } from "../../ppo_unknown/src/consts.ts";
import { ensureTrainer, trainerAct } from "./trainer.ts";

const WARMUP = 20;
const ITERS = 200;

const out = document.getElementById("out") as HTMLPreElement;
const gpuSpan = document.getElementById("gpu") as HTMLSpanElement;
const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>(".bar button"));

function log(line = "") {
  out.textContent += line + "\n";
}

type Stats = { mean: number; p50: number; p90: number; p99: number; min: number; max: number };
function summarize(xs: number[]): Stats {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  const pct = (p: number) => s[Math.min(n - 1, Math.round(p * (n - 1)))];
  return {
    mean: s.reduce((a, b) => a + b, 0) / n,
    p50: pct(0.5),
    p90: pct(0.9),
    p99: pct(0.99),
    min: s[0],
    max: s[n - 1],
  };
}

const pad = (s: string | number, w: number) => String(s).padStart(w);

/** Categorical sample from a probability row (CPU), same role as Rust's sample_categorical. */
function sampleCategorical(probs: Float32Array | Int32Array | Uint8Array): number {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i] as number;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ── tfjs: full act()-equivalent at batch N (forward + softmax + readback + per-row sample) ──
async function benchTf(policy: tf.LayersModel, value: tf.LayersModel, batch = 1, iters = ITERS) {
  const board = tf.randomUniform([batch, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS], -1, 1);
  const cmask = tf.ones([batch, BOARD_CELLS]); // all board cells valid
  const addMask = tf.zeros([batch, ACTION_DIM_TOTAL]); // additive action mask: 0 = all allowed

  const once = async () => {
    const logits = policy.predict([board, cmask]) as tf.Tensor;
    const v = value.predict([board, cmask]) as tf.Tensor;
    const probs = tf.softmax(tf.add(logits, addMask)); // masked softmax over the 43 actions
    const [p, vv] = await Promise.all([probs.data(), v.data()]); // force GPU completion
    for (let i = 0; i < batch; i++) {
      sampleCategorical(
        (p as Float32Array).subarray(i * ACTION_DIM_TOTAL, (i + 1) * ACTION_DIM_TOTAL),
      );
    }
    void vv;
    logits.dispose();
    v.dispose();
    probs.dispose();
  };

  for (let i = 0; i < WARMUP; i++) await once();

  const samples: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t = performance.now();
    await once();
    samples.push(performance.now() - t);
  }
  board.dispose();
  cmask.dispose();
  addMask.dispose();
  return summarize(samples);
}

// ── Burn: act_batch(N) — one forward + one readback for the whole batch ──
async function benchBurnBatch(batch: number, iters = ITERS) {
  const trainer = await ensureTrainer();
  const boards = new Float32Array(batch * BOARD_CELLS * BOARD_CHANNELS);
  for (let i = 0; i < boards.length; i++) boards[i] = Math.random() * 2 - 1;
  const masks = new Float32Array(batch * ACTION_DIM_TOTAL); // 0 = all allowed

  for (let i = 0; i < WARMUP; i++) await trainer.act_batch(boards, masks, batch, false);

  const samples: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t = performance.now();
    await trainer.act_batch(boards, masks, batch, false);
    samples.push(performance.now() - t);
  }
  return summarize(samples);
}

// ── Burn: N single-inferences SEQUENTIAL vs PARALLEL (overlap the readback gaps) ──
// Uses bench_forwards(board,mask,1) — a `&self` method, so concurrent in-flight calls
// are allowed (act_batch is `&mut self` and would panic on overlap). If each call is
// mostly an idle wait for the readback, firing them concurrently lets those waits
// overlap → N inferences finish in ~one wait instead of N. This is the throughput
// ceiling when work pipelines instead of serializing.
async function benchBurnSeqVsParallel(n: number) {
  const board = new Float32Array(BOARD_CELLS * BOARD_CHANNELS);
  for (let i = 0; i < board.length; i++) board[i] = Math.random() * 2 - 1;
  const mask = new Float32Array(ACTION_DIM_TOTAL);

  // warmup both paths
  for (let i = 0; i < 10; i++) await trainer.bench_forwards(board, mask, 1);
  await Promise.all(Array.from({ length: 16 }, () => trainer.bench_forwards(board, mask, 1)));

  const tSeq = performance.now();
  for (let i = 0; i < n; i++) await trainer.bench_forwards(board, mask, 1);
  const seq = performance.now() - tSeq;

  const tPar = performance.now();
  await Promise.all(Array.from({ length: n }, () => trainer.bench_forwards(board, mask, 1)));
  const par = performance.now() - tPar;

  return { seq, par, n };
}

// ── tfjs: N single-inferences SEQUENTIAL vs PARALLEL (same shape as the Burn probe) ──
async function benchTfSeqVsParallel(policy: tf.LayersModel, value: tf.LayersModel, n: number) {
  const board = tf.randomUniform([1, BOARD_ROWS, BOARD_COLS, BOARD_CHANNELS], -1, 1);
  const cmask = tf.ones([1, BOARD_CELLS]);
  const addMask = tf.zeros([1, ACTION_DIM_TOTAL]);

  const once = async () => {
    const logits = policy.predict([board, cmask]) as tf.Tensor;
    const v = value.predict([board, cmask]) as tf.Tensor;
    const masked = tf.add(logits, addMask);
    const probs = tf.softmax(masked);
    const [p] = await Promise.all([probs.data(), v.data()]);
    sampleCategorical(p as Float32Array);
    logits.dispose();
    v.dispose();
    masked.dispose();
    probs.dispose();
  };

  for (let i = 0; i < 10; i++) await once();
  await Promise.all(Array.from({ length: 16 }, () => once()));

  const tSeq = performance.now();
  for (let i = 0; i < n; i++) await once();
  const seq = performance.now() - tSeq;

  const tPar = performance.now();
  await Promise.all(Array.from({ length: n }, () => once()));
  const par = performance.now() - tPar;

  board.dispose();
  cmask.dispose();
  addMask.dispose();
  return { seq, par, n };
}

// ── Burn: act() single-tank (forward + masked softmax + sample + readback) ──
async function benchBurn() {
  await ensureTrainer();
  const board = new Float32Array(BOARD_CELLS * BOARD_CHANNELS);
  for (let i = 0; i < board.length; i++) board[i] = Math.random() * 2 - 1;
  const mask = new Float32Array(ACTION_DIM_TOTAL); // 0 = all actions allowed

  for (let i = 0; i < WARMUP; i++) await trainerAct(board, mask, false);

  const samples: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t = performance.now();
    await trainerAct(board, mask, false);
    samples.push(performance.now() - t);
  }
  return summarize(samples);
}

function row(label: string, s: Stats) {
  return `${label.padEnd(18)} ${pad(s.p50.toFixed(3), 9)} ${pad(s.p90.toFixed(3), 9)} ${pad(
    s.p99.toFixed(3),
    9,
  )} ${pad(s.min.toFixed(3), 9)} ${pad(s.mean.toFixed(3), 9)}`;
}

// ── lazy shared setup (run once, memoized across button clicks) ─────────────
let setupPromise: Promise<{ policy: tf.LayersModel; value: tf.LayersModel } | null> | undefined;
let lastBurnSingleP50: number | undefined; // for the worker comparison line

function setup() {
  if (setupPromise) return setupPromise;
  setupPromise = (async () => {
    if (!("gpu" in navigator)) {
      log("⚠ navigator.gpu missing — this browser has no WebGPU. Use Chrome/Edge.");
      return null;
    }
    log("Initialising TensorFlow.js WebGPU backend…");
    await tf.setBackend("webgpu");
    await tf.ready();
    const backend = tf.getBackend();
    gpuSpan.textContent = `  ·  tfjs backend: ${backend}`;
    if (backend !== "webgpu") log(`⚠ tfjs fell back to "${backend}" (not webgpu)!`);
    log("Building v4 policy + value networks (tfjs) + Burn V4Trainer (wasm)…\n");
    const policy = createPolicyNetwork();
    const value = createValueNetwork();
    await ensureTrainer();
    return { policy, value };
  })();
  return setupPromise;
}

/** Run one test handler with all buttons disabled; setup() is done first. */
async function withSetup(
  fn: (nets: { policy: tf.LayersModel; value: tf.LayersModel }) => Promise<void>,
) {
  buttons.forEach((b) => (b.disabled = true));
  try {
    const nets = await setup();
    if (nets) await fn(nets);
    log("\n— done —\n");
  } catch (err) {
    log(`\n✗ error: ${String(err)}\n`);
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

// ── test 1: single decision, batch 1, identical work ───────────────────────
async function testSingle({ policy, value }: { policy: tf.LayersModel; value: tf.LayersModel }) {
  log("═══ Single decision, batch = 1 — IDENTICAL work both sides ═══");
  log("    (forward policy+value → masked softmax → readback probs+value → CPU sample)\n");
  log(
    `${"".padEnd(18)} ${pad("p50 ms", 9)} ${pad("p90 ms", 9)} ${pad("p99 ms", 9)} ${pad("min ms", 9)} ${pad("mean ms", 9)}`,
  );
  const tfStats = await benchTf(policy, value);
  log(row("tfjs (JS→webgpu)", tfStats));
  const burnStats = await benchBurn();
  log(row("burn (wasm→webgpu)", burnStats));
  lastBurnSingleP50 = burnStats.p50;
  const ratio = burnStats.p50 / tfStats.p50;
  log(
    `\n→ burn p50 / tfjs p50 = ${ratio.toFixed(2)}×  (${ratio >= 1 ? "burn slower" : "burn faster"})`,
  );
}

// ── test 2: batched sweep ───────────────────────────────────────────────────
async function testBatch({ policy, value }: { policy: tf.LayersModel; value: tf.LayersModel }) {
  log("═══ Batched inference — per-CALL ms and per-SAMPLE ms (lower = better) ═══");
  log(
    `${"batch".padEnd(8)} ${pad("tfjs call", 11)} ${pad("tfjs/samp", 11)}   ${pad("burn call", 11)} ${pad("burn/samp", 11)}   ${pad("burn/tfjs", 9)}`,
  );
  for (const batch of [1, 8, 32, 100]) {
    const iters = batch >= 100 ? 50 : ITERS;
    const tfs = await benchTf(policy, value, batch, iters);
    const bns = await benchBurnBatch(batch, iters);
    const tfPer = tfs.p50 / batch;
    const bnPer = bns.p50 / batch;
    log(
      `${String(batch).padEnd(8)} ${pad(tfs.p50.toFixed(2), 11)} ${pad(tfPer.toFixed(3), 11)}   ${pad(
        bns.p50.toFixed(2),
        11,
      )} ${pad(bnPer.toFixed(3), 11)}   ${pad((bnPer / tfPer).toFixed(2) + "×", 9)}`,
    );
  }
}

// ── test 3: sequential vs parallel (in-flight overlap), both backends ───────
async function testParallel({ policy, value }: { policy: tf.LayersModel; value: tf.LayersModel }) {
  log("═══ N single-inferences SEQUENTIAL vs PARALLEL (in-flight overlap) ═══");
  const line = (tag: string, r: { seq: number; par: number; n: number }) =>
    log(
      `  ${tag.padEnd(6)} n=${String(r.n).padEnd(4)} seq ${r.seq.toFixed(1).padStart(7)} ms (${(r.seq / r.n).toFixed(2)} ms/ea)   par ${r.par
        .toFixed(1)
        .padStart(
          7,
        )} ms (${(r.par / r.n).toFixed(2)} ms/ea)   speedup ${(r.seq / r.par).toFixed(1)}×`,
    );
  for (const n of [50, 200]) {
    line("tfjs", await benchTfSeqVsParallel(policy, value, n));
    line("burn", await benchBurnSeqVsParallel(n));
  }
}

// ── test 4: same probe inside a Web Worker (no rAF) ─────────────────────────
async function testWorker() {
  log("═══ Burn in a Web Worker — NO rAF (production path) ═══");
  const worker = new Worker(new URL("./benchWorker.ts", import.meta.url), { type: "module" });
  const res = await new Promise<{
    error?: string;
    single?: { p50: number; min: number; mean: number };
    batch?: { p50: number };
    batchN?: number;
  }>((resolve) => {
    worker.onmessage = (e) => resolve(e.data);
    worker.onerror = (e) => resolve({ error: e.message });
    worker.postMessage("run");
  });
  worker.terminate();
  if (res.error || !res.single || !res.batch) {
    log(`  worker error: ${res.error ?? "no result"}`);
    return;
  }
  log(
    `  single act():     p50=${res.single.p50.toFixed(2)} ms  min=${res.single.min.toFixed(2)} ms  mean=${res.single.mean.toFixed(2)} ms`,
  );
  log(
    `  act_batch(${res.batchN}):   p50=${res.batch.p50.toFixed(2)} ms  → per-sample ${(res.batch.p50 / (res.batchN ?? 1)).toFixed(3)} ms`,
  );
  if (lastBurnSingleP50 !== undefined) {
    log(
      `  ↳ vs main-thread single act ${lastBurnSingleP50.toFixed(2)} ms (run test 1 first to compare)`,
    );
  }
}

const bind = (id: string, fn: () => void) =>
  document.getElementById(id)?.addEventListener("click", fn);

bind("b-single", () => void withSetup((nets) => testSingle(nets)));
bind("b-batch", () => void withSetup((nets) => testBatch(nets)));
bind("b-parallel", () => void withSetup((nets) => testParallel(nets)));
bind("b-worker", () => void withSetup(() => testWorker()));
document.getElementById("b-clear")?.addEventListener("click", () => {
  out.textContent = "";
});
