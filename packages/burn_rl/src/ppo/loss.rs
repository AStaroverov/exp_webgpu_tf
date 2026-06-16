//! Tensor-level PPO loss operations, generic over the backend so they run under
//! `Autodiff<Wgpu>` during training.
//!
//! NOTE: burn's reductions (`.mean()`, `.sum()`) yield a rank-1 tensor of length
//! one rather than a rank-0 scalar, so scalar losses are typed `Tensor<B, 1>`
//! here (the spec's `Tensor<B, 0>` does not exist in burn 0.21).
//!
//! Numerics mirror the TS reference and `sample.rs`:
//!   * softmax subtracts the row max before `exp` (log-sum-exp stabilisation);
//!   * probabilities are clamped to `[PROB_FLOOR, 1.0]` before any `log` so the
//!     log-prob / entropy stay finite even on (almost) fully-masked rows.
//!
//! The combined action vector is a SINGLE 43-way categorical (see `sample.rs`),
//! so the "sum across heads" in the spec collapses to one log-prob over the full
//! [`crate::constants::ACTION_DIM_TOTAL`] distribution.

use crate::ppo::sample::apply_action_mask;
use burn::tensor::{backend::Backend, Int, Tensor};

/// Lower clamp applied to softmax outputs so a subsequent `log` is finite even
/// for (almost) fully-masked entries. Matches `sample::PROB_FLOOR`.
const PROB_FLOOR: f32 = 1e-8;

/// Numerically-stable softmax over the last dim of a `[batch, action_dim]`
/// tensor, clamped to `[PROB_FLOOR, 1.0]`. The `logits` are expected to already
/// carry the additive action mask (apply [`apply_action_mask`] first), so no
/// separate mask argument is taken here.
fn stable_softmax<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2> {
    let max_ = logits.clone().max_dim(1); // [batch, 1], broadcasts over dim 1
    let exp = (logits - max_).exp();
    let sum_exp = exp.clone().sum_dim(1); // [batch, 1]
    (exp / sum_exp).clamp(PROB_FLOOR, 1.0)
}

/// Gather the per-row log-prob of the taken `actions` from a probability tensor.
///
/// `probs`: `[batch, action_dim]`. `actions`: `[batch, 1]` action indices.
/// Returns `[batch, 1]` log-probs.
fn gather_log_prob<B: Backend>(
    probs: Tensor<B, 2>,
    actions: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    // gather along the action dim (1) using the [batch, 1] index tensor.
    let chosen = probs.gather(1, actions); // [batch, 1]
    chosen.log()
}

/// Clipped PPO surrogate policy loss.
///
/// `logits`: `[batch, action_dim]` RAW new-policy logits. `mask`: `[batch,
/// action_dim]` additive action mask (`0`/[`MASK_NEG`]); it is fused via the
/// canonical [`apply_action_mask`] (Hold force-unmasked) so the new-policy
/// distribution matches the behaviour distribution that produced `old_logp`.
/// `actions`: `[batch, 1]` taken action indices. `old_logp`: `[batch, 1]`
/// behaviour-policy log-probs. `advantages`: `[batch, 1]`.
///
/// `loss = -mean(min(r·A, clamp(r, 1-e, 1+e)·A))` with `r = exp(new_logp - old_logp)`.
pub fn clipped_surrogate_loss<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    actions: Tensor<B, 2, Int>,
    old_logp: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
    clip: f32,
) -> Tensor<B, 1> {
    let probs = stable_softmax(apply_action_mask(logits, mask));
    let new_logp = gather_log_prob(probs, actions); // [batch, 1]

    // r = exp(new_logp - old_logp)
    let ratio = (new_logp - old_logp).exp(); // [batch, 1]

    let unclipped = ratio.clone() * advantages.clone();
    let clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages;

    // -mean(min(r·A, clip(r)·A))
    let surrogate = unclipped.min_pair(clipped); // elementwise min, [batch, 1]
    -surrogate.mean()
}

/// Clipped value loss.
///
/// `delta = (v - old_v).clamp(-c, c); loss = mean(max((ret - v)^2, (ret - old_v - delta)^2)) * vf_coef`.
pub fn clipped_value_loss<B: Backend>(
    values: Tensor<B, 2>,
    old_values: Tensor<B, 2>,
    returns: Tensor<B, 2>,
    clip: f32,
    vf_coef: f32,
) -> Tensor<B, 1> {
    // delta = clamp(v - old_v, -c, c)
    let delta = (values.clone() - old_values.clone()).clamp(-clip, clip);
    let v_clipped = old_values + delta; // old_v + delta

    let loss_unclipped = (returns.clone() - values).powf_scalar(2.0);
    let loss_clipped = (returns - v_clipped).powf_scalar(2.0);

    let loss = loss_unclipped.max_pair(loss_clipped); // elementwise max
    loss.mean() * vf_coef
}

/// Masked categorical entropy averaged over batch and heads.
///
/// `entropy = -mean_{batch,heads} sum_k prob_k · log(prob_k)` where `prob` is
/// the masked softmax of the canonically-masked logits (Hold force-unmasked, via
/// [`apply_action_mask`]). The combined action is a single categorical, so
/// "heads" collapses to the one distribution; the per-row entropy is
/// `-sum_k prob_k·log(prob_k)` and we average over the batch. Using the identical
/// masked distribution as the surrogate keeps entropy consistent with `new_logp`.
pub fn masked_entropy<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    // masked_k = logit_k + mask_k (Hold force-unmasked), then stable softmax.
    let probs = stable_softmax(apply_action_mask(logits, mask));
    // per-row entropy H = -sum_k p·log p, then mean over batch.
    let plogp = probs.clone() * probs.log(); // [batch, action_dim]
    let row_entropy = -plogp.sum_dim(1); // [batch, 1]
    row_entropy.mean()
}

/// Schulman approximate KL: `kl = mean(exp(logr) - 1 - logr)`, with `logr`
/// already clamped by the caller. Reported as a detached scalar diagnostic.
pub fn schulman_kl<B: Backend>(log_ratio: Tensor<B, 2>) -> Tensor<B, 1> {
    let ratio = log_ratio.clone().exp();
    let kl = ratio - log_ratio - 1.0; // exp(logr) - logr - 1
    kl.mean()
}
