//! RMSNorm: `x / sqrt(mean(x²) + eps) * weight`, with a learnable scale (gain)
//! per feature and NO bias.
//!
//! Matches the TF.js v4 `RMSNormLayer`
//! (`packages/ppo/src/models/Layers/RMSNormLayer.ts`):
//!
//! ```text
//! RMS(x) = sqrt(mean(x², last_axis) + epsilon)   // epsilon = 1e-6
//! output = (x / RMS(x)) * scale
//! ```
//!
//! The TF.js layer has a learnable `scale` (gain) only and NO bias, so this layer
//! carries no bias term either — a zero-init, decay-exempt bias would just be a
//! free parameter that can drift with no counterpart on the TF.js side.

use burn::module::{Module, Param};
use burn::tensor::{backend::Backend, Tensor};

/// v4 RMSNorm epsilon — the `RMSNormLayer` default (`1e-6`).
pub const RMSNORM_EPS: f64 = 1e-6;

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    /// Build an RMSNorm over `dim` features. `weight` (the gain / scale) inits to
    /// `1`. Pass [`RMSNORM_EPS`] for the v4-faithful epsilon.
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    /// `x`: `[batch, tokens, dim]` → normalised tensor of the same shape.
    ///
    /// `mean(x²)` is taken over the last axis (`dim`) and broadcast back across
    /// it; the result is scaled by `weight` (`[dim]`, broadcast over `batch` and
    /// `tokens`). There is no bias term (matches TF.js `RMSNormLayer`).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dim = x.dims()[2];

        // RMS(x) = sqrt(mean(x², last_axis) + eps), shape [batch, tokens, 1].
        let mean_sq = x.clone().powf_scalar(2.0).mean_dim(2);
        let rms = mean_sq.add_scalar(self.eps).sqrt();

        // x / RMS(x); RMS broadcasts over the last axis.
        let normalized = x / rms;

        // Per-feature scale: [dim] reshaped to [1, 1, dim] to broadcast.
        let weight = self.weight.val().reshape([1, 1, dim]);

        normalized * weight
    }
}
