//! Native inference performance benchmarks for the v4 Perceiver actor-critic.
//!
//! Two angles, both on the native WebGPU backend (Metal on macOS):
//!
//!   1. `inference_forward_throughput` — raw `ActorCriticV4::forward` (board +
//!      content mask → logits/value) across a range of batch sizes. Reports the
//!      per-call latency distribution and samples/sec, so batch scaling and the
//!      fixed per-dispatch overhead are both visible.
//!   2. `inference_act_latency` — the full single-tank `V4Trainer::act` path
//!      (forward + masked softmax + categorical sample + GPU readback), the
//!      latency a live decision driver actually pays per tank per tick.
//!
//! Both are `#[ignore]`d so the normal `cargo test` gate stays fast. Run with:
//!
//! ```sh
//! cargo test --test perf -- --ignored --nocapture
//! ```
//!
//! GPU work is async; every timed iteration forces completion by reading the
//! output tensors back (`into_data`), so the measured time includes the kernel
//! execution, not just the dispatch. A warmup phase absorbs runtime init,
//! autotune, and shape-specific kernel compilation before timing begins.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Tensor;

use burn_rl::test_support::{
    ActorCriticV4, TestBackend, ACTION_DIM_TOTAL, BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS,
    BOARD_ROWS,
};
use burn_rl::V4Trainer;

type B = TestBackend;

const WARMUP: usize = 20;

/// Deterministic pseudo-random buffer of `len` values in `[-1, 1]` (LCG, no dep).
fn rand_data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_board(batch: usize, device: &WgpuDevice) -> Tensor<B, 4> {
    let data = rand_data(batch * BOARD_CELLS * BOARD_CHANNELS, 0xBEEF);
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
        batch,
        BOARD_ROWS,
        BOARD_COLS,
        BOARD_CHANNELS,
    ])
}

/// Content mask `[batch, 121]`, all cells valid — the worst case for the
/// cross-attention (every board token is read).
fn make_content_mask(batch: usize, device: &WgpuDevice) -> Tensor<B, 2> {
    let data = vec![1.0f32; batch * BOARD_CELLS];
    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([batch, BOARD_CELLS])
}

struct Stats {
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    min: f64,
    max: f64,
}

fn summarize(mut xs: Vec<f64>) -> Stats {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let pct = |p: f64| xs[((p * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    Stats {
        mean: xs.iter().sum::<f64>() / n as f64,
        p50: pct(0.50),
        p90: pct(0.90),
        p99: pct(0.99),
        min: xs[0],
        max: xs[n - 1],
    }
}

#[test]
#[ignore = "perf benchmark; run with: cargo test --test perf -- --ignored --nocapture"]
fn inference_forward_throughput() {
    let device = WgpuDevice::default();
    let model = ActorCriticV4::<B>::new(&device);

    println!("\nv4 ActorCriticV4::forward — native wgpu (Metal)");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "batch", "p50 ms", "p90 ms", "p99 ms", "min ms", "samp/s"
    );

    for &batch in &[1usize, 8, 32, 128] {
        let board = make_board(batch, &device);
        let mask = make_content_mask(batch, &device);

        // Warmup: runtime init + autotune + shape-specific kernel compile.
        for _ in 0..WARMUP {
            let (logits, value) = model.forward(board.clone(), mask.clone());
            let _ = logits.into_data();
            let _ = value.into_data();
        }

        let iters = 100;
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let (logits, value) = model.forward(board.clone(), mask.clone());
            // Force GPU completion by reading both heads back.
            let _ = logits.into_data();
            let _ = value.into_data();
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let s = summarize(samples);
        let throughput = batch as f64 / (s.p50 / 1e3);
        println!(
            "{:>6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>12.0}",
            batch, s.p50, s.p90, s.p99, s.min, throughput
        );
    }
    println!();
}

#[test]
#[ignore = "perf benchmark; run with: cargo test --test perf -- --ignored --nocapture"]
fn inference_act_latency() {
    let device = WgpuDevice::default();
    let mut trainer = V4Trainer::new_with_device(device, 99);

    let board = rand_data(BOARD_CELLS * BOARD_CHANNELS, 0x1234);
    let mask = vec![0.0f32; ACTION_DIM_TOTAL]; // all actions allowed

    for _ in 0..WARMUP {
        let _ = pollster::block_on(trainer.act(&board, &mask, false));
    }

    let iters = 200;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = pollster::block_on(trainer.act(&board, &mask, false));
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let s = summarize(samples);
    println!("\nv4 act() single-tank latency — forward + masked softmax + sample + readback");
    println!(
        "  mean={:.3}ms p50={:.3}ms p90={:.3}ms p99={:.3}ms min={:.3}ms max={:.3}ms  ({:.0} act/s @ p50)\n",
        s.mean, s.p50, s.p90, s.p99, s.min, s.max, 1e3 / s.p50
    );
}
