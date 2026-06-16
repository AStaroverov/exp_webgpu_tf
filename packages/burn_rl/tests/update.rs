//! Native tests for `V4Trainer::update` — the PPO/Retrace learner step (lib.rs).
//!
//! The smoke test asserts `update` is finite over a few iterations; these add the
//! two properties a finiteness check can't see:
//!
//!   1. `update_is_finite_and_sane_over_iters` — many updates on one trainer stay
//!      finite and in-range (losses, entropy ≤ ln(43), kl ≥ 0, lr > 0). Catches
//!      blow-ups / NaNs that only appear after several optimiser steps.
//!   2. `update_moves_value_toward_returns` — feeding a constant positive-reward
//!      rollout drives the critic's value estimate UP. This is the end-to-end
//!      "the gradient actually flows and the optimiser actually steps" gate:
//!      forward → loss → backward → AdamW → changed weights, observed through the
//!      public `act` API (which returns the value estimate).
//!
//! Native WebGPU (Metal on macOS), runnable without a browser:
//!
//! ```sh
//! cargo test --test update -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use burn::backend::wgpu::WgpuDevice;
use burn_rl::test_support::{ACTION_DIM_TOTAL, BOARD_CELLS, BOARD_CHANNELS, MASK_NEG};
use burn_rl::V4Trainer;

/// Tiny LCG so rollouts are deterministic without a dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_f32() * n as f32) as usize % n
    }
}

struct Rollout {
    boards: Vec<f32>,
    masks: Vec<f32>,
    actions: Vec<i32>,
    old_logp: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
}

/// Random rollout with valid masks (Hold always allowed), correct obs shape.
fn synth_rollout(steps: usize, seed: u64) -> Rollout {
    let mut rng = Lcg(seed | 1);
    let mut r = Rollout {
        boards: Vec::with_capacity(steps * BOARD_CELLS * BOARD_CHANNELS),
        masks: Vec::with_capacity(steps * ACTION_DIM_TOTAL),
        actions: Vec::with_capacity(steps),
        old_logp: Vec::with_capacity(steps),
        rewards: Vec::with_capacity(steps),
        dones: Vec::with_capacity(steps),
        values: Vec::with_capacity(steps),
    };
    for t in 0..steps {
        for _ in 0..(BOARD_CELLS * BOARD_CHANNELS) {
            r.boards.push(rng.next_f32() * 2.0 - 1.0);
        }
        let mut row = [0.0f32; ACTION_DIM_TOTAL];
        let mut allowed = vec![0usize];
        for k in 1..ACTION_DIM_TOTAL {
            if rng.next_f32() < 0.7 {
                allowed.push(k);
            } else {
                row[k] = MASK_NEG;
            }
        }
        r.masks.extend_from_slice(&row);
        let a = allowed[rng.below(allowed.len())];
        r.actions.push(a as i32);
        r.old_logp.push((1.0f32 / allowed.len() as f32).ln());
        r.rewards.push(rng.next_f32() - 0.5);
        r.dones
            .push(if (t + 1) % 8 == 0 || t == steps - 1 { 1.0 } else { 0.0 });
        r.values.push(rng.next_f32() * 2.0 - 1.0);
    }
    r
}

fn run_update(trainer: &mut V4Trainer, r: &Rollout) -> burn_rl::V4IterStats {
    pollster::block_on(trainer.update(
        &r.boards,
        &r.masks,
        &r.actions,
        &r.old_logp,
        &r.rewards,
        &r.dones,
        &r.values,
    ))
}

#[test]
fn update_is_finite_and_sane_over_iters() {
    // NOTE: cross-instance determinism is intentionally NOT asserted — the
    // `V4Trainer` seed only seeds the action sampler, NOT weight init (Burn's
    // default Initializer draws from getrandom), so two instances start from
    // different weights. Here we drive ONE trainer through several updates and
    // assert every stat stays finite and in a sane range.
    let mut trainer = V4Trainer::new_with_device(WgpuDevice::default(), 2024);

    for iter in 0..6 {
        let r = synth_rollout(64, 0x51 ^ (iter as u64 * 2654435761));
        let s = run_update(&mut trainer, &r);

        assert!(s.policy_loss.is_finite(), "iter {iter}: policy_loss not finite");
        assert!(s.value_loss.is_finite(), "iter {iter}: value_loss not finite");
        assert!(s.entropy.is_finite(), "iter {iter}: entropy not finite");
        assert!(s.kl.is_finite() && s.grad_norm.is_finite() && s.lr.is_finite());

        assert!(s.entropy >= -1e-4, "iter {iter}: entropy negative: {}", s.entropy);
        assert!(
            s.entropy <= (ACTION_DIM_TOTAL as f32).ln() + 1e-3,
            "iter {iter}: entropy above ln(43): {}",
            s.entropy
        );
        assert!(s.value_loss >= -1e-4, "iter {iter}: value_loss negative: {}", s.value_loss);
        assert!(s.kl >= -1e-3, "iter {iter}: kl meaningfully negative: {}", s.kl);
        assert!(s.grad_norm >= 0.0 && s.lr > 0.0);
    }
}

#[test]
fn update_moves_value_toward_returns() {
    // Feeding a constant board with all-positive rewards must drive the critic's
    // value estimate for that board UP — the end-to-end gradient/optimiser gate.
    let mut trainer = V4Trainer::new_with_device(WgpuDevice::default(), 7);

    // Fixed observation + all-allowed mask (additive 0). Action 0 = Hold.
    let board = vec![0.05f32; BOARD_CELLS * BOARD_CHANNELS];
    let mask = vec![0.0f32; ACTION_DIM_TOTAL];

    let value_before = pollster::block_on(trainer.act(&board, &mask, true))[2];

    // Rollout: same board every step, Hold, reward +1, behaviour value 0.
    let steps = 64;
    let mut r = Rollout {
        boards: Vec::new(),
        masks: Vec::new(),
        actions: vec![0i32; steps],
        old_logp: vec![0.0f32; steps],
        rewards: vec![1.0f32; steps],
        dones: (0..steps)
            .map(|t| if (t + 1) % 16 == 0 || t == steps - 1 { 1.0 } else { 0.0 })
            .collect(),
        values: vec![0.0f32; steps],
    };
    for _ in 0..steps {
        r.boards.extend_from_slice(&board);
        r.masks.extend_from_slice(&mask);
    }

    let mut last = value_before;
    for _ in 0..40 {
        let s = run_update(&mut trainer, &r);
        assert!(s.value_loss.is_finite(), "value_loss became non-finite");
        last = s.value_loss; // touched to keep the value meaningful in failure msgs
    }
    let _ = last;

    let value_after = pollster::block_on(trainer.act(&board, &mask, true))[2];

    assert!(
        value_after > value_before + 1e-3,
        "critic did not learn: value_before={value_before} value_after={value_after}"
    );
}
