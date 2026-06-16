//! PPO + Retrace training configuration and the CPU/tensor ops that make up the
//! update. The submodules split by concern: advantage estimation (`gae`,
//! `retrace`), tensor loss ops (`loss`), masked sampling (`sample`), gradient
//! clipping (`clip`), and the KL-adaptive LR schedule (`lr`).

pub mod clip;
pub mod gae;
pub mod loss;
pub mod lr;
pub mod retrace;
pub mod sample;

// Re-exports for convenience at the `ppo::` path.
pub use clip::apply_global_norm_clip;
pub use gae::{compute_gae, normalize};
pub use loss::{
    clipped_surrogate_loss, clipped_value_loss, masked_entropy, schulman_kl,
};
pub use lr::KLScheduler;
pub use retrace::compute_retrace;
#[allow(unused_imports)]
pub use retrace::RetraceOutput;
pub use sample::{apply_action_mask, masked_softmax};
#[allow(unused_imports)]
pub use sample::sample_categorical;

// `optim` lives at the crate root (`crate::optim`), not under `ppo`, because the
// AdamW + decay-exemption helpers are not advantage/loss ops. Re-exported here so
// the full PPO update surface is reachable from the `ppo::` path per the spec.
pub use crate::optim::{init_adamw_pair, split_grads_by_decay};
#[allow(unused_imports)]
pub use crate::optim::{adamw_decayed_config, adamw_exempt_config, is_decay_exempt};

/// PPO + Retrace hyper-parameters. Defaults mirror the TypeScript reference.
#[derive(Clone, Copy, Debug)]
pub struct PpoConfig {
    // --- rollout / optimisation loop ---
    pub steps: usize,  // environment steps collected per training iteration
    pub epochs: usize, // optimisation passes over each rollout

    // --- advantage estimation ---
    pub gamma: f32,  // discount factor
    pub lambda: f32, // GAE / Retrace smoothing

    // --- PPO clipping ---
    pub clip_policy: f32,
    pub clip_value: f32,

    // --- loss weights ---
    pub ent_coef: f32,
    pub vf_coef: f32,

    // --- gradient clipping ---
    pub grad_clip_norm: f32,

    // --- AdamW ---
    pub adamw_lr: f64,
    pub adamw_beta1: f64,
    pub adamw_beta2: f64,
    pub adamw_eps: f64,
    pub adamw_wd: f64,

    // --- KL-adaptive LR schedule ---
    pub kl_history_size: usize,
    pub kl_high: f64,
    pub kl_low: f64,
    pub kl_mult_high: f64,
    pub kl_mult_low: f64,

    // --- numerics ---
    pub log_ratio_clamp: (f32, f32),
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            steps: 1024,
            epochs: 4,
            gamma: 0.99,
            lambda: 0.95,
            clip_policy: 0.2,
            clip_value: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            grad_clip_norm: 5.0,
            adamw_lr: 2.5e-3,
            adamw_beta1: 0.9,
            adamw_beta2: 0.999,
            adamw_eps: 1e-5,
            adamw_wd: 1e-6,
            kl_history_size: 25,
            kl_high: 0.02,
            kl_low: 0.005,
            kl_mult_high: 0.95,
            kl_mult_low: 1.05,
            log_ratio_clamp: (-20.0, 20.0),
        }
    }
}
