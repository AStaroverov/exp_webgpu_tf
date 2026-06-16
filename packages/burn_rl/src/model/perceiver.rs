//! Perceiver block — a faithful Rust/Burn port of the TF.js v4 reference
//! (`packages/ppo/src/models/Layers/PerceiverLayer.ts` +
//! `ApplyLayers.ts::applyCrossAttentionLayer` / `applySelfTransformerLayer`).
//!
//! Per depth iteration the v4 block does (exactly):
//!
//! ```text
//! // 1. cross-attention (PRE-NORM: applyPerceiverLayer is called with
//! //    preNorm=true, so applyCrossAttentionLayer RMSNorms the query latents
//! //    (QNorm) and the key/value tokens (KVNorm) BEFORE attention). Latents
//! //    attend to the board tokens with the board mask on the key/value axis.
//! //    The residual adds back the RAW latents (the pre-norm input).
//! qNorm    = RMSNorm(latents)                          // QNorm  (preNorm=true)
//! kvNorm   = RMSNorm(tokens)                            // KVNorm (preNorm=true)
//! crossOut = MHA(q = qNorm, kv = kvNorm, kvMask = mask)
//! x        = latents + crossOut                       // residual (raw latents)
//!
//! // 2. self-attention transformer over the latents (applySelfTransformerLayer):
//! //    pre-LN self-attention (QNorm inside attention), then a gated FFN.
//! selfNorm = RMSNorm(x)                                // QNorm (preNorm=true)
//! selfOut  = MHA(q = selfNorm, kv = selfNorm)          // self-attn, no mask
//! x        = x + selfOut                               // attnResidual
//! ffnNorm  = RMSNorm(x)                                // ln2
//! gate     = sigmoid(Linear_4d(ffnNorm))               // SiLU-style gate
//! up       = Linear_4d(ffnNorm)
//! ffnOut   = Linear_d(up * gate)
//! x        = x + ffnOut                                // ffnAdd
//! ```
//!
//! Masking: the TF.js `MultiHeadAttentionLayer` uses a multiplicative-valid mask
//! (`1` = keep, `0` = drop) turned into the additive term `(kvMask - 1) * 1e9`.
//! Burn's [`MultiHeadAttention`] instead takes a boolean `mask_pad` where `true`
//! marks padded (dropped) keys and fills those scores with `min_float`. We build
//! the bool pad mask as `board_mask == 0` and set `min_float = MASK_NEG` so the
//! numerics match. The query-side mask in v4 is a `MaskLikeLayer` (all ones) for
//! both the cross and self steps, so it is a no-op and is intentionally dropped.
//!
//! All `Linear`/attention layers carry no behaviour beyond what Burn provides; the
//! block is pure data + a deterministic forward. `N` blocks are stacked by
//! `model/mod.rs` (4 for the policy branch, 2 for the value branch — separate
//! instances per the spec, never a shared/looped module).

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::{backend::Backend, Bool, Tensor};

use crate::constants::MASK_NEG;
use crate::model::norm::{RMSNorm, RMSNORM_EPS};

/// FFN inner-dimension multiplier (v4 uses `dModel * 4`).
const FFN_MULT: usize = 4;

#[derive(Module, Debug)]
pub struct PerceiverBlock<B: Backend> {
    /// Cross-attention: latents (query) attend to board tokens (key/value).
    pub cross_attn: MultiHeadAttention<B>,
    /// Pre-norm on the query latents before cross-attention (the v4 `QNorm`).
    pub cross_q_norm: RMSNorm<B>,
    /// Pre-norm on the key/value tokens before cross-attention (v4 `KVNorm`).
    pub cross_kv_norm: RMSNorm<B>,
    /// Self-attention over the latents.
    pub self_attn: MultiHeadAttention<B>,
    /// Pre-norm applied to the latents before self-attention (the v4 `QNorm`).
    pub self_norm: RMSNorm<B>,
    /// Pre-norm before the FFN (the v4 `ln2`).
    pub ffn_norm: RMSNorm<B>,
    /// Gate projection `d -> 4d`, `sigmoid` (no bias).
    pub ffn_gate: Linear<B>,
    /// Up projection `d -> 4d`, linear (no bias).
    pub ffn_up: Linear<B>,
    /// Down projection `4d -> d`, linear (no bias).
    pub ffn_down: Linear<B>,
}

/// Construction config for a single [`PerceiverBlock`].
#[derive(Config, Debug)]
pub struct PerceiverBlockConfig {
    /// Model / token feature dimension (`256` policy, `128` value).
    pub dim: usize,
    /// Number of attention heads (`8` policy, `4` value).
    pub heads: usize,
}

impl PerceiverBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PerceiverBlock<B> {
        // Attention with dropout disabled (training is full-batch PPO, the v4
        // reference uses no attention dropout) and `min_float = MASK_NEG` so the
        // masked-key fill matches the TF.js `(kvMask - 1) * 1e9` additive term.
        let attn = |dim: usize, heads: usize| {
            MultiHeadAttentionConfig::new(dim, heads)
                .with_dropout(0.0)
                .with_min_float(MASK_NEG as f64)
                .init::<B>(device)
        };

        let inner = self.dim * FFN_MULT;
        // FFN projections are bias-free (v4: `useBias: false`).
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(false).init::<B>(device);

        PerceiverBlock {
            cross_attn: attn(self.dim, self.heads),
            cross_q_norm: RMSNorm::new(self.dim, RMSNORM_EPS, device),
            cross_kv_norm: RMSNorm::new(self.dim, RMSNORM_EPS, device),
            self_attn: attn(self.dim, self.heads),
            self_norm: RMSNorm::new(self.dim, RMSNORM_EPS, device),
            ffn_norm: RMSNorm::new(self.dim, RMSNORM_EPS, device),
            ffn_gate: lin(self.dim, inner),
            ffn_up: lin(self.dim, inner),
            ffn_down: lin(inner, self.dim),
        }
    }
}

impl<B: Backend> PerceiverBlock<B> {
    /// `latents`: `[batch, n_latents, dim]`; `tokens`: `[batch, n_tokens, dim]`;
    /// `mask`: `[batch, n_tokens]` (1 = valid key, 0 = masked).
    /// Returns updated latents `[batch, n_latents, dim]`.
    pub fn forward(
        &self,
        latents: Tensor<B, 3>,
        tokens: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // --- 1. cross-attention (pre-LN QNorm/KVNorm) + residual ------------
        // preNorm=true: RMSNorm the query latents (QNorm) and the key/value
        // tokens (KVNorm) before attention; the residual adds back the RAW
        // latents (pre-norm input), mirroring the self-attention pattern below.
        // Burn pad mask: `true` marks dropped keys → board cells with mask == 0.
        let q_norm = self.cross_q_norm.forward(latents.clone());
        let kv_norm = self.cross_kv_norm.forward(tokens);
        let pad_mask: Tensor<B, 2, Bool> = mask.equal_elem(0.0);
        let cross_out = self
            .cross_attn
            .forward(
                MhaInput::new(q_norm, kv_norm.clone(), kv_norm).mask_pad(pad_mask),
            )
            .context;
        let x = latents + cross_out;

        // --- 2. self-attention (pre-LN QNorm) + residual --------------------
        let self_norm = self.self_norm.forward(x.clone());
        let self_out = self
            .self_attn
            .forward(MhaInput::self_attn(self_norm))
            .context;
        let x = x + self_out;

        // --- 3. gated FFN (pre-LN ln2) + residual ---------------------------
        let ffn_norm = self.ffn_norm.forward(x.clone());
        let gate = sigmoid(self.ffn_gate.forward(ffn_norm.clone()));
        let up = self.ffn_up.forward(ffn_norm);
        let ffn_out = self.ffn_down.forward(up * gate);

        x + ffn_out
    }
}
