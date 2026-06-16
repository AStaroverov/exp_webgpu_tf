//! Masked categorical sampling for the combined 43-dim action vector.
//!
//! `masked_softmax`: additive masking — `masked_k = logit_k + mask_k` where a
//! forbidden action carries the additive [`MASK_NEG`] (`-1e9`) in `mask`. Two
//! properties are guaranteed:
//!   * **Hold is never masked.** The [`HOLD_ACTION`] (index 0) column is forced
//!     to an additive `0` regardless of what the caller passes, so a valid
//!     fallback action always exists.
//!   * **Finite even when a row is (almost) fully masked.** We subtract the row
//!     max before `exp` (standard log-sum-exp stabilisation); because Hold is
//!     always unmasked the row max is always finite, the masked entries
//!     underflow toward `0`, and the output is clamped to `[1e-8, 1.0]` so any
//!     downstream `log` stays finite.
//!
//! `sample_categorical`: builds the CDF (`cumsum` of probs) and picks
//! `argmin(CDF >= r)` with `r ~ U[0,1)` drawn from the xorshift [`Rng`]. The
//! sample step reads the probabilities back to the CPU (`into_data`) and uses
//! the shared [`Rng::sample_categorical`] primitive so it stays bit-for-bit
//! reproducible from a seed. `greedy = true` selects the argmax instead.

use crate::constants::{HOLD_ACTION, MASK_NEG};
use crate::rng::Rng;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

/// Lower clamp applied to softmax outputs so a subsequent `log` is finite even
/// for fully-masked entries.
const PROB_FLOOR: f32 = 1e-8;

/// Apply the additive action mask to raw logits — the SINGLE canonical place the
/// mask is fused, shared by sampling (`masked_softmax`), the policy/value loss,
/// the entropy term, and the KL diagnostic. Returns `logits + mask` with the
/// [`HOLD_ACTION`] column force-unmasked (its additive mask set to `0`).
///
/// `logits`/`mask` are `[batch, ACTION_DIM_TOTAL]`; `mask` is additive (`0`
/// allowed, [`MASK_NEG`] forbidden). Because the behaviour log-probs collected at
/// rollout time come from `masked_softmax` (Hold-never-masked), every training
/// quantity must use this identical masked distribution so `old_logp`,
/// `new_logp`, and the entropy are all over the same action set.
pub fn apply_action_mask<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, action_dim] = logits.dims();
    let device = logits.device();
    let mask = unmask_hold(mask, batch, action_dim, &device);
    logits + mask
}

/// Numerically-stable masked softmax over the last dim. `logits`/`mask` are
/// `[batch, ACTION_DIM_TOTAL]`; returns `[batch, ACTION_DIM_TOTAL]`
/// probabilities clamped to `[1e-8, 1.0]`.
///
/// `mask` is additive: `0` for a permitted action, [`MASK_NEG`] for a forbidden
/// one. The [`HOLD_ACTION`] column is forced to `0` here so Hold is never
/// masked.
pub fn masked_softmax<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    // masked_k = logit_k + mask_k (Hold force-unmasked) — same canonical fuse.
    let masked = apply_action_mask(logits, mask);

    // Subtract the row max for numerical stability. Because Hold is unmasked the
    // row max is always finite, so masked-out entries become large-negative and
    // exp() underflows to ~0 instead of producing NaN/Inf.
    let max_ = masked.clone().max_dim(1); // [batch, 1], broadcasts over dim 1
    let exp = (masked - max_).exp();
    let sum_exp = exp.clone().sum_dim(1); // [batch, 1]
    let probs = exp / sum_exp;

    // Clamp so a downstream log() is finite even when a row is mostly masked.
    probs.clamp(PROB_FLOOR, 1.0)
}

/// Sample one action per batch row over the full [`crate::constants::ACTION_DIM_TOTAL`]
/// action vector. Returns `(actions [batch], logp [batch])`.
///
/// `logits`/`mask` are `[batch, ACTION_DIM_TOTAL]`. `greedy = true` takes the
/// argmax (deterministic inference); otherwise the action is drawn from the
/// masked categorical via the xorshift `rng`.
///
/// NOTE: this reads the probabilities back to the CPU synchronously
/// (`into_data`) to drive the reproducible CPU sampler. On the wasm/WebGPU
/// target prefer the async readback path in `lib.rs` (`into_data_async` →
/// [`Rng::sample_categorical`]); a blocking readback can deadlock there. This
/// sync form is kept for the spec signature and native/test use.
pub fn sample_categorical<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    rng: &mut Rng,
    greedy: bool,
) -> (Tensor<B, 1, Int>, Tensor<B, 1>) {
    let [batch, action_dim] = logits.dims();
    let device = logits.device();

    let probs = masked_softmax(logits, mask);

    // Read probabilities back to the CPU for the reproducible categorical draw.
    let probs_cpu: Vec<f32> = probs
        .into_data()
        .iter::<f32>()
        .collect();

    let mut actions: Vec<i32> = Vec::with_capacity(batch);
    let mut logps: Vec<f32> = Vec::with_capacity(batch);

    for b in 0..batch {
        let row = &probs_cpu[b * action_dim..(b + 1) * action_dim];

        let action = if greedy {
            // argmax over the row.
            let mut best = HOLD_ACTION;
            for k in 1..action_dim {
                if row[k] > row[best] {
                    best = k;
                }
            }
            best
        } else {
            // CDF = cumsum(probs); action = argmin(CDF >= r), r ~ U[0,1).
            rng.sample_categorical(row)
        };

        actions.push(action as i32);
        logps.push(row[action].max(PROB_FLOOR).ln());
    }

    let actions_t =
        Tensor::<B, 1, Int>::from_data(TensorData::new(actions, [batch]), &device);
    let logp_t = Tensor::<B, 1>::from_data(TensorData::new(logps, [batch]), &device);
    (actions_t, logp_t)
}

/// Force the additive mask of the [`HOLD_ACTION`] column to `0` so Hold is never
/// masked. Returns the mask with column `0` overwritten with zeros.
fn unmask_hold<B: Backend>(
    mask: Tensor<B, 2>,
    batch: usize,
    action_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    debug_assert!(
        HOLD_ACTION < action_dim,
        "HOLD_ACTION out of range for action_dim {action_dim}"
    );
    let _ = MASK_NEG; // documents the additive-mask convention this guards against.
    let zeros = Tensor::<B, 2>::zeros([batch, 1], device);
    mask.slice_assign([0..batch, HOLD_ACTION..HOLD_ACTION + 1], zeros)
}
