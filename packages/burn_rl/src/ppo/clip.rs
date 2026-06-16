//! Global-norm gradient clipping over a whole [`GradientsParams`] container.
//!
//! ```text
//! norm = sqrt( Σ_p Σ_i g_{p,i}² )      (sum of squares over EVERY gradient element of EVERY param)
//! coef = min(1, clipnorm / norm)
//! g'   = coef · g                       (scale every gradient by the same coef)
//! ```
//!
//! This is the TF/PyTorch `clip_by_global_norm` semantics used by the TS
//! reference — a SINGLE norm taken across the concatenation of all gradients,
//! NOT burn's built-in [`burn::grad_clipping::GradientClipping::Norm`], which
//! clips each tensor by its own per-tensor norm. We therefore hand-build the
//! reduction with a module visitor.
//!
//! ## API / backend notes (burn 0.21)
//!
//! * burn's `GradientsParams` is NOT generic over the backend — its *methods*
//!   are. The container is keyed by [`ParamId`] and stores each gradient on the
//!   *inner* (non-autodiff) backend `B::InnerBackend`, exactly as
//!   `GradientsParamsConverter` registers them. So the public signature takes
//!   the autodiff module `&M` (to enumerate `(ParamId, rank D)` via a
//!   `ModuleVisitor`) alongside the `&mut GradientsParams`. This differs from
//!   the spec's `GradientsParams<B>` — that type does not exist in burn 0.21.
//!
//! * Gradient tensors have heterogeneous static ranks (1..=4 here). A
//!   `ModuleVisitor` is the only rank-generic entry point burn exposes: its
//!   `visit_float::<D>` carries the static rank, letting us `get`/`remove`/
//!   `register` at the matching rank `B::InnerBackend`.
//!
//! * The actual scaling stays fully on-device (`tensor * coef`); the ONLY
//!   readback is the single scalar global norm, which is why this is `async`
//!   and uses `into_data_async` — matching the crate-wide rule of never doing a
//!   blocking `Tensor -> Vec` conversion on the wgpu backend (the spec's sync
//!   `-> f32` would block / panic on wasm-wgpu).

use burn::module::{AutodiffModule, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

/// Accumulates `Σ g²` across every gradient tensor referenced by a module's
/// float params, on-device, into a single running `[1]` tensor.
struct SumSqVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    /// Running sum of squares on the inner backend, or `None` until the first
    /// gradient is seen (we cannot build a zero scalar without a device handle).
    acc: Option<Tensor<B::InnerBackend, 1>>,
}

impl<B> ModuleVisitor<B> for SumSqVisitor<'_, B>
where
    B: AutodiffBackend,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // Gradients are stored on the inner backend, keyed by param id and at
        // the param's static rank D.
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            // Σ_i g_i² for this tensor, reduced to a single scalar `[1]`.
            let sumsq: Tensor<B::InnerBackend, 1> = grad.powf_scalar(2.0).sum();
            self.acc = Some(match self.acc.take() {
                Some(prev) => prev + sumsq,
                None => sumsq,
            });
        }
    }
}

/// Scales every gradient tensor in the container by the same scalar `coef`,
/// in place (remove → scale → re-register at the matching rank).
struct ScaleVisitor<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    coef: f32,
    _backend: core::marker::PhantomData<B>,
}

impl<B> ModuleVisitor<B> for ScaleVisitor<'_, B>
where
    B: AutodiffBackend,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads
                .register::<B::InnerBackend, D>(param.id, grad.mul_scalar(self.coef));
        }
    }
}

/// Clip all gradients in `grads` so their GLOBAL L2 norm is at most `clipnorm`.
///
/// * `grads`  — the gradient container produced by
///   [`GradientsParams::from_grads`].
/// * `module` — the autodiff module the gradients belong to; used only to
///   enumerate the `(ParamId, rank)` pairs (the container is otherwise opaque
///   to rank).
/// * `clipnorm` — the maximum allowed global norm (PPO `grad_clip_norm`, =5.0
///   per [`crate::ppo::PpoConfig`]).
///
/// Returns the pre-clip global norm so the caller can log / health-check it
/// (e.g. guard against a NaN norm before stepping the optimiser). If the norm
/// is non-finite or below `clipnorm` the gradients are left untouched (`coef`
/// would be `1`), so a NaN does not get multiplied through silently — the
/// caller should inspect the returned value.
pub async fn apply_global_norm_clip<B, M>(
    grads: &mut GradientsParams,
    module: &M,
    clipnorm: f32,
) -> f32
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    // --- pass 1: Σ g² across all gradients, on-device ---
    let mut sumsq_visitor = SumSqVisitor::<B> {
        grads,
        acc: None,
    };
    // `visit` dispatches `visit_float::<D>` per float param at its static rank.
    module.visit(&mut sumsq_visitor);

    let Some(sumsq) = sumsq_visitor.acc else {
        // No float gradients at all — nothing to clip.
        return 0.0;
    };

    // sqrt(Σ g²); single-scalar readback (the only one in this routine).
    let norm = sumsq
        .sqrt()
        .into_data_async()
        .await
        .unwrap()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0);

    // coef = min(1, clipnorm / norm). Skip scaling when nothing to do or when
    // the norm is degenerate (0 / NaN / inf) — never propagate a NaN coef.
    if !norm.is_finite() || norm <= clipnorm || norm == 0.0 {
        return norm;
    }
    let coef = clipnorm / norm; // strictly < 1 here

    // --- pass 2: scale every gradient by coef, on-device ---
    let mut scale_visitor = ScaleVisitor::<B> {
        grads,
        coef,
        _backend: core::marker::PhantomData,
    };
    module.visit(&mut scale_visitor);

    norm
}
