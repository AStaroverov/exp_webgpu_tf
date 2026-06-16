/**
 * benchWorker — runs the SAME Burn act() / act_batch() latency probe as bench.ts,
 * but inside a Web Worker. Workers have NO requestAnimationFrame, so this isolates
 * whether the ~14ms single-inference stall seen on the main thread is the rAF-gated
 * readback (main-thread only) or something intrinsic. If single-act p50 here is far
 * below the main-thread 14ms, the stall was rAF.
 */

import init, { V4Trainer } from "../../burn_rl/pkg/burn_rl.js";
import { BOARD_CELLS, BOARD_CHANNELS } from "../../ppo_unknown/src/models/dims.ts";
import { ACTION_DIM_TOTAL } from "../../ppo_unknown/src/consts.ts";

function summarize(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  const pct = (p: number) => s[Math.min(n - 1, Math.round(p * (n - 1)))];
  return { p50: pct(0.5), p90: pct(0.9), min: s[0], mean: s.reduce((a, b) => a + b, 0) / n };
}

self.onmessage = async (e: MessageEvent) => {
  if (e.data !== "run") return;
  try {
    if (!("gpu" in navigator)) {
      (self as unknown as Worker).postMessage({ error: "navigator.gpu missing in worker" });
      return;
    }
    await init();
    const trainer = await V4Trainer.create(42);

    const board = new Float32Array(BOARD_CELLS * BOARD_CHANNELS);
    for (let i = 0; i < board.length; i++) board[i] = Math.random() * 2 - 1;
    const mask = new Float32Array(ACTION_DIM_TOTAL);

    // single act()
    for (let i = 0; i < 20; i++) await trainer.act(board, mask, false);
    const single: number[] = [];
    for (let i = 0; i < 200; i++) {
      const t = performance.now();
      await trainer.act(board, mask, false);
      single.push(performance.now() - t);
    }

    // batched act_batch(N)
    const N = 100;
    const boards = new Float32Array(N * BOARD_CELLS * BOARD_CHANNELS);
    for (let i = 0; i < boards.length; i++) boards[i] = Math.random() * 2 - 1;
    const masks = new Float32Array(N * ACTION_DIM_TOTAL);
    for (let i = 0; i < 10; i++) await trainer.act_batch(boards, masks, N, false);
    const batch: number[] = [];
    for (let i = 0; i < 50; i++) {
      const t = performance.now();
      await trainer.act_batch(boards, masks, N, false);
      batch.push(performance.now() - t);
    }

    (self as unknown as Worker).postMessage({
      single: summarize(single),
      batch: summarize(batch),
      batchN: N,
    });
  } catch (err) {
    (self as unknown as Worker).postMessage({ error: String(err) });
  }
};
