//! PPO in Burn, in the browser.
//!
//! A self-contained test of the full RL loop running on the WebGPU backend
//! compiled to wasm: rollout collection, GAE, clipped-surrogate PPO updates with
//! autodiff + Adam, and inference. Drives a tiny grid-world environment.
//!
//! Everything async-reads tensors back from the GPU (`into_data_async`) because
//! the synchronous readbacks deadlock under wasm.

mod constants;
mod env;
mod model;
mod optim;
mod ppo;
mod record;
mod rng;

use burn::backend::wgpu::{
    init_setup_async, MemoryConfiguration, RuntimeOptions, Wgpu, WgpuDevice,
};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{activation, Int, Tensor, TensorData};
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

use constants::{ACTION_DIM_TOTAL, BOARD_CELLS, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS};
use env::{GridWorld, N_ACTIONS};
use model::{ActorCritic, ActorCriticConfig, ActorCriticV4};
use optim::collect_decay_exempt_ids;
use ppo::{
    apply_global_norm_clip, clipped_surrogate_loss, clipped_value_loss, compute_retrace,
    init_adamw_pair, masked_entropy, masked_softmax, schulman_kl, split_grads_by_decay,
    KLScheduler, PpoConfig,
};
use rng::Rng;

/// Inference backend (plain WebGPU) and training backend (autodiff on top).
type InferBackend = Wgpu;
type TrainBackend = Autodiff<Wgpu>;
type Opt = OptimizerAdaptor<Adam, ActorCritic<TrainBackend>, TrainBackend>;

/// AdamW optimiser pair (decayed / exempt) over the v4 network.
type V4Opt = OptimizerAdaptor<burn::optim::AdamW, ActorCriticV4<TrainBackend>, TrainBackend>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// wgpu runtime options for the trainers.
///
/// The default (`MemoryConfiguration::SubSlices`) builds every memory pool with
/// `dealloc_period: None` (== `u64::MAX`), so GPU pages are **never** released —
/// freed slices are only reused *within* already-allocated pages. Our rollout
/// batch size varies every iteration (we update on the first productive episode),
/// so each update produces tensor shapes that don't fit existing free slices and
/// forces fresh pages that then accumulate forever (observed: ~66 GB RSS).
///
/// `ExclusivePages` instead assigns each size-bucket a finite `dealloc_period`
/// (~5000 allocations, scaled by size), so pages unused for a while are reclaimed
/// and total memory stays bounded regardless of shape churn.
fn wgpu_runtime_options() -> RuntimeOptions {
    RuntimeOptions {
        memory_config: MemoryConfiguration::ExclusivePages,
        ..Default::default()
    }
}

/// Internal surface exposed ONLY for the native correctness tests
/// (`tests/correctness.rs`). Not part of the public API — hidden from docs and
/// never used by the wasm bindings. Lets the integration-test crate reach the v4
/// model, the gather layer, the masked softmax, and the relevant constants
/// without making the whole module tree `pub`.
#[doc(hidden)]
#[cfg(not(target_arch = "wasm32"))]
pub mod test_support {
    pub use crate::constants::{
        ACTION_CELL_INDEXES, ACTION_CELL_INDEXES_LEN, ACTION_DIM_TOTAL, BOARD_CELLS,
        BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, HOLD_ACTION, MASK_NEG,
    };
    pub use crate::model::hex::HexRingGather;
    pub use crate::model::ActorCriticV4;
    pub use crate::ppo::masked_softmax;

    /// Inference backend used by the tests (plain WebGPU, no autodiff).
    pub type TestBackend = burn::backend::wgpu::Wgpu;
}

/// Stats returned to JS after each training iteration.
#[wasm_bindgen]
pub struct IterStats {
    pub avg_return: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub episodes: u32,
    pub steps: u32,
}

#[wasm_bindgen]
pub struct Trainer {
    device: WgpuDevice,
    model: ActorCritic<TrainBackend>,
    optim: Opt,
    env: GridWorld,
    rng: Rng,
    cfg: PpoConfig,
    obs_dim: usize,
}

#[wasm_bindgen]
impl Trainer {
    /// Async constructor: initialises the WebGPU device, then builds the network
    /// and optimiser. In JS: `const t = await Trainer.create(5, 64, 42);`
    pub async fn create(grid_size: usize, hidden: usize, seed: u32) -> Trainer {
        let device = WgpuDevice::default();
        init_setup_async::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
            &device,
            wgpu_runtime_options(),
        )
        .await;

        let env = GridWorld::new(grid_size);
        let obs_dim = env.obs_dim();
        let model = ActorCriticConfig {
            obs: obs_dim,
            hidden,
            actions: N_ACTIONS,
        }
        .init::<TrainBackend>(&device);
        let optim = AdamConfig::new().init();

        Trainer {
            device,
            model,
            optim,
            env,
            rng: Rng::new(seed as u64),
            cfg: PpoConfig::default(),
            obs_dim,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn grid_size(&self) -> usize {
        self.env.size()
    }

    #[wasm_bindgen(getter)]
    pub fn n_actions(&self) -> usize {
        N_ACTIONS
    }

    /// Collect one rollout and run `epochs` PPO updates on it.
    pub async fn train_iteration(&mut self) -> IterStats {
        let n = self.cfg.steps;

        // ---- Rollout (acting uses the non-autodiff "valid" copy of the net) ----
        let infer = self.model.valid();

        let mut obs_buf: Vec<f32> = Vec::with_capacity(n * self.obs_dim);
        let mut actions: Vec<i32> = Vec::with_capacity(n);
        let mut old_logp: Vec<f32> = Vec::with_capacity(n);
        let mut values: Vec<f32> = Vec::with_capacity(n);
        let mut rewards: Vec<f32> = Vec::with_capacity(n);
        let mut dones: Vec<bool> = Vec::with_capacity(n);

        let mut ep_return = 0.0f32;
        let mut ep_returns: Vec<f32> = Vec::new();

        let mut obs = self.env.reset(&mut self.rng);
        for _ in 0..n {
            let (probs, value) = act(&infer, &self.device, &obs).await;
            let action = self.rng.sample_categorical(&probs);
            let logp = probs[action].max(1e-8).ln();

            let step = self.env.step(action);

            obs_buf.extend_from_slice(&obs);
            actions.push(action as i32);
            old_logp.push(logp);
            values.push(value);
            rewards.push(step.reward);
            dones.push(step.done);
            ep_return += step.reward;

            if step.done {
                ep_returns.push(ep_return);
                ep_return = 0.0;
                obs = self.env.reset(&mut self.rng);
            } else {
                obs = self.env.obs();
            }
        }

        // Bootstrap value for the final (possibly non-terminal) state.
        let (_, last_value) = act(&infer, &self.device, &obs).await;

        // ---- Advantages / returns (CPU) ----
        let (mut adv, returns) = ppo::compute_gae(
            &rewards,
            &values,
            &dones,
            last_value,
            self.cfg.gamma,
            self.cfg.lambda,
        );
        ppo::normalize(&mut adv);

        // ---- Tensors for the update ----
        let dev = &self.device;
        let obs_t = Tensor::<TrainBackend, 1>::from_floats(obs_buf.as_slice(), dev)
            .reshape([n, self.obs_dim]);
        let act_t =
            Tensor::<TrainBackend, 1, Int>::from_ints(actions.as_slice(), dev).reshape([n, 1]);
        let old_logp_t =
            Tensor::<TrainBackend, 1>::from_floats(old_logp.as_slice(), dev).reshape([n, 1]);
        let adv_t = Tensor::<TrainBackend, 1>::from_floats(adv.as_slice(), dev).reshape([n, 1]);
        let ret_t = Tensor::<TrainBackend, 1>::from_floats(returns.as_slice(), dev).reshape([n, 1]);

        // ---- PPO epochs ----
        let mut model = self.model.clone();
        let mut last_pl = 0.0f32;
        let mut last_vl = 0.0f32;
        let mut last_ent = 0.0f32;

        for _ in 0..self.cfg.epochs {
            let (logits, value) = model.forward(obs_t.clone());
            let logp_all = activation::log_softmax(logits.clone(), 1);
            let probs = activation::softmax(logits, 1);

            let new_logp = logp_all.clone().gather(1, act_t.clone());
            let entropy = -(probs * logp_all).sum_dim(1).mean();

            let ratio = (new_logp - old_logp_t.clone()).exp();
            let surr1 = ratio.clone() * adv_t.clone();
            let surr2 = ratio.clamp(1.0 - self.cfg.clip_policy, 1.0 + self.cfg.clip_policy)
                * adv_t.clone();
            // Elementwise min(surr1, surr2): where surr1 > surr2 take surr2.
            let mask = surr1.clone().greater(surr2.clone());
            let policy_loss = surr1.mask_where(mask, surr2).mean().neg();

            let value_loss = (value - ret_t.clone()).powf_scalar(2.0).mean();

            let loss = policy_loss.clone()
                + value_loss.clone().mul_scalar(self.cfg.vf_coef)
                - entropy.clone().mul_scalar(self.cfg.ent_coef);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = self.optim.step(self.cfg.adamw_lr, model, grads);

            last_pl = read_scalar(policy_loss).await;
            last_vl = read_scalar(value_loss).await;
            last_ent = read_scalar(entropy).await;
        }
        self.model = model;

        let episodes = ep_returns.len();
        let avg_return = if episodes > 0 {
            ep_returns.iter().sum::<f32>() / episodes as f32
        } else {
            0.0
        };

        IterStats {
            avg_return,
            policy_loss: last_pl,
            value_loss: last_vl,
            entropy: last_ent,
            episodes: episodes as u32,
            steps: n as u32,
        }
    }

    /// Action probabilities for a single cell (length `n_actions`).
    pub async fn infer(&self, cell: usize) -> Vec<f32> {
        let infer = self.model.valid();
        let obs = self.env.obs_of(cell);
        let (probs, _) = act(&infer, &self.device, &obs).await;
        probs
    }

    /// Greedy (argmax) action for every cell — for visualising the learned policy.
    pub async fn greedy_actions(&self) -> Vec<u32> {
        let (logits, _) = self.forward_all_cells().await;
        let a = N_ACTIONS;
        (0..logits.len() / a)
            .map(|i| {
                let row = &logits[i * a..i * a + a];
                let mut best = 0usize;
                for j in 1..a {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best as u32
            })
            .collect()
    }

    /// Value estimate for every cell — for visualising the value function.
    pub async fn values(&self) -> Vec<f32> {
        let (_, values) = self.forward_all_cells().await;
        values
    }

    /// Forward every cell at once. Returns (flat logits `[cells * n_actions]`,
    /// values `[cells]`).
    async fn forward_all_cells(&self) -> (Vec<f32>, Vec<f32>) {
        let infer = self.model.valid();
        let cells = self.obs_dim; // grid has obs_dim == cell count
        let mut flat = Vec::with_capacity(cells * self.obs_dim);
        for c in 0..cells {
            flat.extend_from_slice(&self.env.obs_of(c));
        }
        let x = Tensor::<InferBackend, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([cells, self.obs_dim]);
        let (logits, value) = infer.forward(x);
        let logits = read_all(logits).await;
        let value = read_all(value).await;
        (logits, value)
    }
}

// ===========================================================================
// v4 training path: external-rollout PPO + Retrace over the Perceiver network.
// ===========================================================================

/// Stats returned to JS after each v4 PPO update over an externally-supplied
/// rollout. All fields are finite scalars read back from the GPU asynchronously.
#[wasm_bindgen]
pub struct V4IterStats {
    /// Mean Retrace return target over the batch (proxy for rollout quality).
    pub avg_return: f32,
    /// Clipped-surrogate policy loss of the final PPO epoch.
    pub policy_loss: f32,
    /// Clipped value loss of the final PPO epoch.
    pub value_loss: f32,
    /// Masked categorical entropy of the final PPO epoch.
    pub entropy: f32,
    /// Schulman approximate KL of the final PPO epoch.
    pub kl: f32,
    /// Learning rate after the KL-adaptive update.
    pub lr: f32,
    /// Pre-clip global gradient norm of the final PPO epoch (health-check).
    pub grad_norm: f32,
}

/// Single-thread v4 PPO trainer.
///
/// Holds the v4 [`ActorCriticV4`] under `Autodiff<Wgpu>`, a decayed/exempt AdamW
/// optimiser pair (see [`crate::optim`]), a KL-adaptive LR scheduler, and the
/// [`PpoConfig`]. The rollout is supplied from JS/TS — this type never generates
/// experience itself; it consumes a flat batch and runs `epochs` PPO updates.
#[wasm_bindgen]
pub struct V4Trainer {
    device: WgpuDevice,
    model: ActorCriticV4<TrainBackend>,
    optim_decayed: V4Opt,
    optim_exempt: V4Opt,
    exempt_ids: HashSet<burn::module::ParamId>,
    kl_sched: KLScheduler,
    cfg: PpoConfig,
    lr: f64,
    rng: Rng,
}

#[wasm_bindgen]
impl V4Trainer {
    /// Async constructor: initialises the WebGPU device, builds the v4 network
    /// and the AdamW optimiser pair. In JS: `const t = await V4Trainer.create(42);`
    pub async fn create(seed: u32) -> V4Trainer {
        let device = WgpuDevice::default();
        init_setup_async::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
            &device,
            wgpu_runtime_options(),
        )
        .await;
        V4Trainer::new_with_device(device, seed)
    }

    /// Number of actions in the combined action vector (43).
    #[wasm_bindgen(getter)]
    pub fn action_dim(&self) -> usize {
        ACTION_DIM_TOTAL
    }

    /// Run `epochs` PPO updates over one externally-supplied rollout batch and
    /// return the iteration stats.
    ///
    /// Buffer layout (all flat, row-major, length `steps` unless noted):
    /// * `boards`     — `[steps * 11 * 11 * 22]` observations.
    /// * `masks`      — `[steps * 43]` additive action masks (`0` allowed,
    ///   `-1e9` forbidden); Hold (index 0) is force-unmasked internally.
    /// * `actions`    — `[steps]` taken action indices.
    /// * `old_logp`   — `[steps]` behaviour-policy log-probs.
    /// * `rewards`    — `[steps]`.
    /// * `dones`      — `[steps]` (`0`/`1`).
    /// * `values`     — `[steps]` behaviour-policy value estimates.
    pub async fn update(
        &mut self,
        boards: &[f32],
        masks: &[f32],
        actions: &[i32],
        old_logp: &[f32],
        rewards: &[f32],
        dones: &[f32],
        values: &[f32],
    ) -> V4IterStats {
        let n = actions.len();
        assert!(n > 0, "empty rollout");
        assert_eq!(boards.len(), n * BOARD_CELLS * BOARD_CHANNELS, "boards size");
        assert_eq!(masks.len(), n * ACTION_DIM_TOTAL, "masks size");
        assert_eq!(old_logp.len(), n, "old_logp size");
        assert_eq!(rewards.len(), n, "rewards size");
        assert_eq!(dones.len(), n, "dones size");
        assert_eq!(values.len(), n, "values size");

        // ---- Retrace targets (CPU). log_ratio at rollout time is 0 (π_new==μ
        // on the very first epoch); the scan clamps internally regardless. ----
        let dones_b: Vec<bool> = dones.iter().map(|&d| d > 0.5).collect();
        let log_ratios = vec![0.0f32; n];
        let mut retrace = compute_retrace(
            rewards,
            &dones_b,
            values,
            &log_ratios,
            self.cfg.gamma,
            self.cfg.lambda,
        );
        // Match tfjs `computeRetraceTargets`, which returns `normalize(advantages)`
        // (zero-mean / unit-std). The GridWorld GAE path normalizes too (see above).
        ppo::normalize(&mut retrace.advantages);

        let avg_return = retrace.returns.iter().copied().sum::<f32>() / n as f32;

        // ---- Static tensors for the update (built once, reused per epoch). ----
        let dev = &self.device;
        let board_t = Tensor::<TrainBackend, 1>::from_floats(boards, dev).reshape([
            n,
            BOARD_ROWS,
            BOARD_COLS,
            BOARD_CHANNELS,
        ]);
        let mask_t =
            Tensor::<TrainBackend, 1>::from_floats(masks, dev).reshape([n, ACTION_DIM_TOTAL]);
        let act_t = Tensor::<TrainBackend, 1, Int>::from_data(
            TensorData::new(actions.to_vec(), [n]),
            dev,
        )
        .reshape([n, 1]);
        let old_logp_t =
            Tensor::<TrainBackend, 1>::from_floats(old_logp, dev).reshape([n, 1]);
        let old_val_t =
            Tensor::<TrainBackend, 1>::from_floats(values, dev).reshape([n, 1]);
        let adv_t = Tensor::<TrainBackend, 1>::from_floats(retrace.advantages.as_slice(), dev)
            .reshape([n, 1]);
        let ret_t = Tensor::<TrainBackend, 1>::from_floats(retrace.returns.as_slice(), dev)
            .reshape([n, 1]);
        // Board CONTENT mask `[n, 121]` fed to the perceiver cross-attention —
        // distinct from the `[n, 43]` ACTION mask applied to the logits. Derived
        // from the boards exactly as the tfjs reference does outside the graph
        // (`state/InputTensors.ts`): a cell is "content" iff ANY channel other
        // than the always-on CoordX/CoordY planes is non-zero.
        let content_mask = derive_content_mask(boards, n);
        let board_mask_t =
            Tensor::<TrainBackend, 1>::from_floats(content_mask.as_slice(), dev)
                .reshape([n, BOARD_CELLS]);

        let mut last_pl = 0.0f32;
        let mut last_vl = 0.0f32;
        let mut last_ent = 0.0f32;
        let mut last_kl = 0.0f32;
        let mut last_norm = 0.0f32;
        // One Schulman-KL sample per PPO epoch — collected and fed to the KL
        // scheduler all at once after the epoch loop, mirroring tfjs
        // `klList.push(...)` per epoch then `klHistory.add(...klList)`
        // (createPolicyLearnerAgent.ts). The median window is therefore over
        // EPOCHS, not a single per-iteration value.
        let mut kl_list: Vec<f64> = Vec::with_capacity(self.cfg.epochs);

        // burn's `step` consumes-and-returns the module by value; thread it
        // through a local so we never have to fabricate a placeholder.
        let mut model = self.model.clone();

        for _ in 0..self.cfg.epochs {
            // Forward (raw logits; the canonical action mask is fused inside the
            // loss/entropy ops so the whole loss path sees one masked dist).
            let (raw_logits, value) = model.forward(board_t.clone(), board_mask_t.clone());

            let policy_loss = clipped_surrogate_loss(
                raw_logits.clone(),
                mask_t.clone(),
                act_t.clone(),
                old_logp_t.clone(),
                adv_t.clone(),
                self.cfg.clip_policy,
            );
            let value_loss = clipped_value_loss(
                value,
                old_val_t.clone(),
                ret_t.clone(),
                self.cfg.clip_value,
                self.cfg.vf_coef,
            );
            let entropy = masked_entropy(raw_logits.clone(), mask_t.clone());

            // total = policy + value*vf(applied in loss) - ent*ent_coef
            let loss = policy_loss.clone() + value_loss.clone()
                - entropy.clone().mul_scalar(self.cfg.ent_coef);

            // ---- Backward + global-norm clip + decayed/exempt AdamW step. ----
            let grads = loss.backward();
            let mut grads = GradientsParams::from_grads(grads, &model);
            last_norm =
                apply_global_norm_clip::<TrainBackend, _>(&mut grads, &model, self.cfg.grad_clip_norm)
                    .await;

            let (decayed, exempt) =
                split_grads_by_decay::<TrainBackend, _>(grads, &model, &self.exempt_ids);
            model = self.optim_decayed.step(self.lr, model, decayed);
            model = self.optim_exempt.step(self.lr, model, exempt);

            // ---- Diagnostics: new-policy KL under the just-updated weights. ----
            let new_logp = current_logp(&model, &board_t, &board_mask_t, &mask_t, &act_t);
            let log_ratio = new_logp - old_logp_t.clone();
            let log_ratio = log_ratio.clamp(self.cfg.log_ratio_clamp.0, self.cfg.log_ratio_clamp.1);
            let kl = schulman_kl(log_ratio);

            last_pl = read_scalar(policy_loss).await;
            last_vl = read_scalar(value_loss).await;
            last_ent = read_scalar(entropy).await;
            last_kl = read_scalar(kl).await;
            kl_list.push(last_kl as f64);
        }
        self.model = model;

        // ---- KL-adaptive LR for the next iteration. ----
        // Feed EVERY epoch's KL into the ring buffer (mirrors tfjs
        // `klHistory.add(...klList)`), then drive the LR off the median of the
        // whole window. `extend` pushes all samples; `adjust` applies the
        // multiplier + clamp using the post-extend median.
        self.kl_sched.extend(kl_list);
        if let Some(median_kl) = self.kl_sched.median() {
            self.lr = self.kl_sched.adjust(median_kl, self.lr);
        }

        V4IterStats {
            avg_return,
            policy_loss: last_pl,
            value_loss: last_vl,
            entropy: last_ent,
            kl: last_kl,
            lr: self.lr as f32,
            grad_norm: last_norm,
        }
    }

    /// Inference / act: forward a single observation, masked-softmax sample one
    /// action, return `(action, log_prob, value)` packed as `[action, logp, value]`.
    ///
    /// `board` is `[11*11*22]`, `mask` is `[43]` additive. `greedy = true` takes
    /// the argmax instead of sampling.
    pub async fn act(&mut self, board: &[f32], mask: &[f32], greedy: bool) -> Vec<f32> {
        assert_eq!(board.len(), BOARD_CELLS * BOARD_CHANNELS, "board size");
        assert_eq!(mask.len(), ACTION_DIM_TOTAL, "mask size");
        let infer = self.model.valid();
        let dev = &self.device;

        let board_t = Tensor::<InferBackend, 1>::from_floats(board, dev).reshape([
            1,
            BOARD_ROWS,
            BOARD_COLS,
            BOARD_CHANNELS,
        ]);
        let mask_t = Tensor::<InferBackend, 1>::from_floats(mask, dev).reshape([1, ACTION_DIM_TOTAL]);
        // Board CONTENT mask `[1, 121]` for the perceiver, derived from the board
        // exactly as the tfjs reference (`state/InputTensors.ts`): content iff any
        // non-Coord channel is non-zero. Distinct from the `[1, 43]` action mask.
        let content_mask = derive_content_mask(board, 1);
        let board_mask_t = Tensor::<InferBackend, 1>::from_floats(content_mask.as_slice(), dev)
            .reshape([1, BOARD_CELLS]);

        let (logits, value) = infer.forward(board_t, board_mask_t);
        let probs = masked_softmax(logits, mask_t); // [1, 43], Hold force-unmasked

        // ONE GPU→CPU readback, not two. Each `into_data_async().await` is a full
        // async map round-trip (submit → map_async → callback on the next event-loop
        // tick); under wasm that fixed cost (~7 ms) dominates this tiny net, so doing
        // probs and value as separate awaits roughly DOUBLES act latency. Pack them
        // into one `[1, 44]` tensor and read it back in a single round-trip.
        let packed = Tensor::cat(vec![probs, value], 1); // [1, ACTION_DIM_TOTAL + 1]
        let packed_cpu = read_all(packed).await;
        let (probs_cpu, value_slice) = packed_cpu.split_at(ACTION_DIM_TOTAL);
        let value = value_slice[0];

        let action = if greedy {
            let mut best = 0usize;
            for k in 1..probs_cpu.len() {
                if probs_cpu[k] > probs_cpu[best] {
                    best = k;
                }
            }
            best
        } else {
            self.rng.sample_categorical(probs_cpu)
        };
        let logp = probs_cpu[action].max(1e-8).ln();
        vec![action as f32, logp, value]
    }

    /// Batched inference: forward `n` observations at once and return `[action,
    /// logp, value]` per sample, packed row-major (`n * 3`). One forward + one
    /// GPU→CPU readback for the whole batch — the per-call fixed overhead (the
    /// ~10 ms inter-submit idle gap seen at batch 1) is paid ONCE for all `n`
    /// tanks instead of per tank.
    ///
    /// `boards` is `[n * 11 * 11 * BOARD_CHANNELS]`, `masks` is `[n * 43]` additive.
    pub async fn act_batch(&mut self, boards: &[f32], masks: &[f32], n: usize, greedy: bool) -> Vec<f32> {
        assert_eq!(boards.len(), n * BOARD_CELLS * BOARD_CHANNELS, "boards size");
        assert_eq!(masks.len(), n * ACTION_DIM_TOTAL, "masks size");
        let infer = self.model.valid();
        let dev = &self.device;

        let board_t = Tensor::<InferBackend, 1>::from_floats(boards, dev).reshape([
            n,
            BOARD_ROWS,
            BOARD_COLS,
            BOARD_CHANNELS,
        ]);
        let mask_t =
            Tensor::<InferBackend, 1>::from_floats(masks, dev).reshape([n, ACTION_DIM_TOTAL]);
        let content_mask = derive_content_mask(boards, n);
        let board_mask_t = Tensor::<InferBackend, 1>::from_floats(content_mask.as_slice(), dev)
            .reshape([n, BOARD_CELLS]);

        let (logits, value) = infer.forward(board_t, board_mask_t);
        let probs = masked_softmax(logits, mask_t); // [n, 43]
        let packed = Tensor::cat(vec![probs, value], 1); // [n, 44]
        let flat = read_all(packed).await; // n * (ACTION_DIM_TOTAL + 1), single readback

        let stride = ACTION_DIM_TOTAL + 1;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let base = i * stride;
            let row = &flat[base..base + ACTION_DIM_TOTAL];
            let v = flat[base + ACTION_DIM_TOTAL];
            let action = if greedy {
                let mut best = 0usize;
                for k in 1..row.len() {
                    if row[k] > row[best] {
                        best = k;
                    }
                }
                best
            } else {
                self.rng.sample_categorical(row)
            };
            let logp = row[action].max(1e-8).ln();
            out.push(action as f32);
            out.push(logp);
            out.push(v);
        }
        out
    }

    /// Benchmark probe: run `iters` forward passes (policy+value+masked softmax),
    /// chained so none are eliminated, then flush with a SINGLE GPU→CPU readback.
    /// Lets JS separate the two costs:
    ///   * `t(1)`  ≈ one forward + one readback round-trip,
    ///   * `t(50)` ≈ fifty forwards + one readback round-trip.
    /// If `t(50) ≈ t(1)` the per-call cost is the readback latency (compute is
    /// nearly free); if `t(50) ≈ 50·t(1)` it's the kernel compute. Returns a
    /// scalar derived from the result purely to defeat dead-code elimination.
    pub async fn bench_forwards(&self, board: &[f32], mask: &[f32], iters: u32) -> f32 {
        assert_eq!(board.len(), BOARD_CELLS * BOARD_CHANNELS, "board size");
        assert_eq!(mask.len(), ACTION_DIM_TOTAL, "mask size");
        let infer = self.model.valid();
        let dev = &self.device;

        let board_t = Tensor::<InferBackend, 1>::from_floats(board, dev).reshape([
            1,
            BOARD_ROWS,
            BOARD_COLS,
            BOARD_CHANNELS,
        ]);
        let mask_t = Tensor::<InferBackend, 1>::from_floats(mask, dev).reshape([1, ACTION_DIM_TOTAL]);
        let content_mask = derive_content_mask(board, 1);
        let board_mask_t = Tensor::<InferBackend, 1>::from_floats(content_mask.as_slice(), dev)
            .reshape([1, BOARD_CELLS]);

        let mut acc: Option<Tensor<InferBackend, 2>> = None;
        for _ in 0..iters.max(1) {
            let (logits, value) = infer.forward(board_t.clone(), board_mask_t.clone());
            let probs = masked_softmax(logits, mask_t.clone());
            let packed = Tensor::cat(vec![probs, value], 1); // [1, 44]
            acc = Some(match acc {
                Some(a) => a + packed, // chain so the graph isn't dead-code eliminated
                None => packed,
            });
        }
        let out = read_all(acc.unwrap()).await; // SINGLE readback
        out[0]
    }
}

impl V4Trainer {
    /// Build a trainer on an already-initialised device (shared by the wasm
    /// `create` constructor and the native smoke test).
    pub fn new_with_device(device: WgpuDevice, seed: u32) -> V4Trainer {
        let model = ActorCriticV4::<TrainBackend>::new(&device);
        let cfg = PpoConfig::default();
        let (optim_decayed, optim_exempt) = init_adamw_pair::<TrainBackend, ActorCriticV4<TrainBackend>>(
            cfg.adamw_beta1,
            cfg.adamw_beta2,
            cfg.adamw_eps,
            cfg.adamw_wd,
        );
        let exempt_ids = collect_decay_exempt_ids::<TrainBackend, _>(&model);
        // LR-adaptive schedule clamped to a sane band around the base LR.
        let kl_sched = KLScheduler::new(
            cfg.kl_history_size,
            cfg.kl_high,
            cfg.kl_low,
            cfg.kl_mult_high,
            cfg.kl_mult_low,
            cfg.adamw_lr * 1e-2,
            cfg.adamw_lr * 1e2,
        );
        V4Trainer {
            device,
            model,
            optim_decayed,
            optim_exempt,
            exempt_ids,
            kl_sched,
            cfg,
            lr: cfg.adamw_lr,
            rng: Rng::new(seed as u64),
        }
    }
}

/// Per-cell board content mask `[n * BOARD_CELLS]`, derived from a flat
/// `[n * BOARD_CELLS * BOARD_CHANNELS]` board buffer exactly as the tfjs
/// reference does outside the graph (`state/InputTensors.ts`): a cell is
/// "content" (`1.0`) iff ANY channel other than the always-on coordinate planes
/// (CoordX = channel 0, CoordY = channel 1) is non-zero, else `0.0`. Those two
/// planes are written for every in-view cell, so including them would make every
/// cell look like content.
fn derive_content_mask(boards: &[f32], n: usize) -> Vec<f32> {
    // CoordX / CoordY are the first two channels of `BoardChannel` (state/board.ts).
    const COORD_X: usize = 0;
    const COORD_Y: usize = 1;
    let mut out = vec![0.0f32; n * BOARD_CELLS];
    for cell in 0..n * BOARD_CELLS {
        let base = cell * BOARD_CHANNELS;
        let mut has_content = false;
        for ch in 0..BOARD_CHANNELS {
            if ch == COORD_X || ch == COORD_Y {
                continue;
            }
            if boards[base + ch] != 0.0 {
                has_content = true;
                break;
            }
        }
        out[cell] = if has_content { 1.0 } else { 0.0 };
    }
    out
}

/// New-policy log-prob of the taken actions under `model` — for the KL stat.
fn current_logp(
    model: &ActorCriticV4<TrainBackend>,
    board: &Tensor<TrainBackend, 4>,
    board_mask: &Tensor<TrainBackend, 2>,
    action_mask: &Tensor<TrainBackend, 2>,
    actions: &Tensor<TrainBackend, 2, Int>,
) -> Tensor<TrainBackend, 2> {
    let (logits, _) = model.forward(board.clone(), board_mask.clone());
    // Same canonical masking as sampling / the surrogate (Hold force-unmasked).
    let masked = ppo::apply_action_mask(logits, action_mask.clone());
    let logp_all = activation::log_softmax(masked, 1);
    logp_all.gather(1, actions.clone())
}

/// Forward a single observation and return `(action_probs, value)`.
async fn act(
    model: &ActorCritic<InferBackend>,
    device: &WgpuDevice,
    obs: &[f32],
) -> (Vec<f32>, f32) {
    let x = Tensor::<InferBackend, 1>::from_floats(obs, device).reshape([1, obs.len()]);
    let (logits, value) = model.forward(x);
    let probs = activation::softmax(logits, 1);
    let probs = read_all(probs).await;
    let value = read_scalar_infer(value).await;
    (probs, value)
}

async fn read_all<const D: usize>(t: Tensor<InferBackend, D>) -> Vec<f32> {
    t.into_data_async().await.unwrap().iter::<f32>().collect()
}

async fn read_scalar_infer<const D: usize>(t: Tensor<InferBackend, D>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0)
}

async fn read_scalar<const D: usize>(t: Tensor<TrainBackend, D>) -> f32 {
    t.into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0)
}
