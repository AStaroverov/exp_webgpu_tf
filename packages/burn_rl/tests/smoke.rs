//! Native P1 "no-NaN / sane ranges" gate for the v4 PPO trainer.
//!
//! Builds a [`V4Trainer`] on the native WebGPU backend (Metal on macOS),
//! synthesises a tiny random rollout with a valid action mask (Hold always
//! unmasked), runs a handful of `update` iterations feeding the rollout in from
//! the "outside" (exactly as JS/TS would), and asserts every returned statistic
//! is finite and in a sane range. Runnable without a browser:
//!
//! ```sh
//! cargo test --test smoke -- --nocapture
//! ```
//!
//! Only compiled for native targets — the wasm build has no test harness.

#![cfg(not(target_arch = "wasm32"))]

use burn::backend::wgpu::WgpuDevice;
use burn_rl::V4Trainer;

// Mirror the crate constants (the consts module is private).
const BOARD_ROWS: usize = 11;
const BOARD_COLS: usize = 11;
const BOARD_CHANNELS: usize = 19;
const BOARD_CELLS: usize = BOARD_ROWS * BOARD_COLS;
const ACTION_DIM_TOTAL: usize = 43;
const MASK_NEG: f32 = -1e9;

/// Tiny LCG so the rollout is deterministic without pulling in a dep.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_f32() * n as f32) as usize % n
    }
}

/// Build a synthetic rollout of `steps` transitions with valid masks.
struct Rollout {
    boards: Vec<f32>,
    masks: Vec<f32>,
    actions: Vec<i32>,
    old_logp: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<f32>,
    values: Vec<f32>,
}

fn synth_rollout(steps: usize, seed: u64) -> Rollout {
    let mut rng = Lcg(seed | 1);
    let mut boards = Vec::with_capacity(steps * BOARD_CELLS * BOARD_CHANNELS);
    let mut masks = Vec::with_capacity(steps * ACTION_DIM_TOTAL);
    let mut actions = Vec::with_capacity(steps);
    let mut old_logp = Vec::with_capacity(steps);
    let mut rewards = Vec::with_capacity(steps);
    let mut dones = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(steps);

    for t in 0..steps {
        // Random board features in [-1, 1].
        for _ in 0..(BOARD_CELLS * BOARD_CHANNELS) {
            boards.push(rng.next_f32() * 2.0 - 1.0);
        }

        // Mask: Hold (0) always allowed; each other action ~70% allowed.
        let mut row = [0.0f32; ACTION_DIM_TOTAL];
        row[0] = 0.0;
        let mut allowed: Vec<usize> = vec![0];
        for k in 1..ACTION_DIM_TOTAL {
            if rng.next_f32() < 0.7 {
                row[k] = 0.0;
                allowed.push(k);
            } else {
                row[k] = MASK_NEG;
            }
        }
        masks.extend_from_slice(&row);

        // Take a uniformly random *allowed* action; log-prob = ln(1/|allowed|).
        let a = allowed[rng.below(allowed.len())];
        actions.push(a as i32);
        old_logp.push((1.0f32 / allowed.len() as f32).ln());

        rewards.push(rng.next_f32() - 0.5);
        // Terminate every ~8 steps and on the final step.
        dones.push(if (t + 1) % 8 == 0 || t == steps - 1 { 1.0 } else { 0.0 });
        values.push(rng.next_f32() * 2.0 - 1.0);
    }

    Rollout { boards, masks, actions, old_logp, rewards, dones, values }
}

#[test]
fn v4_trainer_no_nan_sane_ranges() {
    // Native wgpu device (Metal on macOS); ops lazily init the runtime.
    let device = WgpuDevice::default();
    let mut trainer = V4Trainer::new_with_device(device, 1234);

    assert_eq!(trainer.action_dim(), ACTION_DIM_TOTAL);

    let steps = 48;

    for iter in 0..5 {
        let r = synth_rollout(steps, 0xABCD ^ (iter as u64 * 7919));
        let stats = pollster::block_on(trainer.update(
            &r.boards,
            &r.masks,
            &r.actions,
            &r.old_logp,
            &r.rewards,
            &r.dones,
            &r.values,
        ));

        eprintln!(
            "iter {iter}: avg_return={:.4} policy_loss={:.4} value_loss={:.4} entropy={:.4} kl={:.5} lr={:.6} grad_norm={:.4}",
            stats.avg_return, stats.policy_loss, stats.value_loss, stats.entropy, stats.kl,
            stats.lr, stats.grad_norm,
        );

        // --- P1 gate: every stat finite ---
        assert!(stats.avg_return.is_finite(), "avg_return not finite: {}", stats.avg_return);
        assert!(stats.policy_loss.is_finite(), "policy_loss not finite: {}", stats.policy_loss);
        assert!(stats.value_loss.is_finite(), "value_loss not finite: {}", stats.value_loss);
        assert!(stats.entropy.is_finite(), "entropy not finite: {}", stats.entropy);
        assert!(stats.kl.is_finite(), "kl not finite: {}", stats.kl);
        assert!(stats.lr.is_finite(), "lr not finite: {}", stats.lr);
        assert!(stats.grad_norm.is_finite(), "grad_norm not finite: {}", stats.grad_norm);

        // --- sane ranges ---
        assert!(stats.entropy >= -1e-4, "entropy negative: {}", stats.entropy);
        // Entropy of a 43-way categorical is bounded by ln(43) ≈ 3.76.
        assert!(stats.entropy <= (ACTION_DIM_TOTAL as f32).ln() + 1e-3, "entropy too large: {}", stats.entropy);
        assert!(stats.value_loss >= -1e-4, "value_loss negative: {}", stats.value_loss);
        assert!(stats.kl >= -1e-3, "kl meaningfully negative: {}", stats.kl);
        assert!(stats.grad_norm >= 0.0, "grad_norm negative: {}", stats.grad_norm);
        assert!(stats.lr > 0.0, "lr non-positive: {}", stats.lr);
    }
}

#[test]
fn v4_trainer_act_is_finite() {
    let device = WgpuDevice::default();
    let mut trainer = V4Trainer::new_with_device(device, 7);

    let board = vec![0.1f32; BOARD_CELLS * BOARD_CHANNELS];
    let mut mask = vec![0.0f32; ACTION_DIM_TOTAL];
    // Mask out everything except Hold and a couple of actions.
    for k in 3..ACTION_DIM_TOTAL {
        mask[k] = MASK_NEG;
    }

    let out = pollster::block_on(trainer.act(&board, &mask, false));
    assert_eq!(out.len(), 3, "act returns [action, logp, value]");
    let (action, logp, value) = (out[0], out[1], out[2]);
    assert!(action.is_finite() && action >= 0.0 && (action as usize) < ACTION_DIM_TOTAL);
    assert!(logp.is_finite() && logp <= 1e-4, "logp must be <= 0: {logp}");
    assert!(value.is_finite());

    // Greedy must also be valid.
    let g = pollster::block_on(trainer.act(&board, &mask, true));
    assert!(g[0].is_finite() && (g[0] as usize) < ACTION_DIM_TOTAL);
}
