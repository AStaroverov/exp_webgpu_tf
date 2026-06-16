//! AdamW with selective weight-decay exemption.
//!
//! Burn's [`AdamW`] (the OSS optimiser) applies a *single* global `weight_decay`
//! to every parameter — it has no per-parameter exemption. The v4 reference, like
//! most transformer setups, excludes **bias**, **normalization scale/bias**, and
//! **σ-style** parameters from decay. That exemption is the only thing hand-built
//! here; AdamW itself is used straight from `burn::optim`.
//!
//! ## How the exemption is implemented
//!
//! We run *two* `AdamW` optimisers that share identical hyper-parameters except
//! for weight decay:
//!
//! * `decayed`  — `weight_decay = adamw_wd` (`1e-6` per spec), for matrix weights.
//! * `exempt`   — `weight_decay = 0`, for bias / norm / σ params.
//!
//! Each PPO step the gradient container ([`GradientsParams`]) is split by
//! [`ParamId`] into a decayed subset and an exempt subset, and the two optimisers
//! step the *same* module in sequence (each ignores the params it has no
//! gradients for). The split key is the parameter's **module path** (e.g.
//! `policy_norm.weight`, `cross_attn.query.bias`), matched by
//! [`is_decay_exempt`].
//!
//! ## Why path-based and not id-based
//!
//! Burn 0.21 gives a leaf [`Param`] no stable string name through the plain
//! `visit_float` callback — only a [`ParamId`]. But the `#[derive(Module)]`
//! codegen wraps every field visit in `enter_module(field_name, …)` /
//! `exit_module(field_name, …)`. So a visitor that maintains a path **stack**
//! (push on enter, pop on exit) knows the field name of whatever `visit_float`
//! fires for. We classify on that path. This is the documented migration risk in
//! the spec ("debug-print actual param names"): if a future burn changes the
//! enter/exit ordering, [`collect_decay_exempt_ids`] / its debug helper is where
//! to look.
//!
//! Constants used (from the spec / [`crate::ppo::PpoConfig`]): `adamw_eps = 1e-5`,
//! `adamw_beta1 = 0.9`, `adamw_beta2 = 0.999`, `adamw_wd = 1e-6`,
//! `adamw_lr = 2.5e-3` (lr is passed at `step` time, not stored on the config).

use std::collections::HashSet;

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param, ParamId};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

/// Returns `true` if the parameter at module path `name` must be **excluded**
/// from weight decay.
///
/// `name` is the parameter's module path (the components joined by `.`, e.g.
/// `policy_perceiver.0.self_norm.weight`). Exempted:
///
/// * any `bias` (Linear / attention biases),
/// * anything under a normalization module (`*norm*`) — both its `weight`
///   (RMSNorm scale `γ`) and `bias`,
/// * `gamma` / `beta` scale-shift params (alternative norm naming),
/// * `sigma` / σ params (e.g. noisy-net style, NoisyLinear `*_sigma`).
///
/// Decayed (returns `false`): plain matrix weights — projections, attention
/// `query/key/value/output` kernels, FFN, the value head, the bias-free scorers.
pub fn is_decay_exempt(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("bias")
        || n.contains("norm")
        || n.contains("gamma")
        || n.contains("beta")
        || n.contains("sigma")
}

// ---------------------------------------------------------------------------
// AdamW config builders.
// ---------------------------------------------------------------------------

/// Build the **decayed** AdamW config (matrix weights): the spec hyper-params
/// with `weight_decay = wd`.
///
/// `lr` is intentionally NOT part of the config — burn's `AdamW` takes the
/// learning rate per `step`, which is what the KL-adaptive schedule
/// ([`crate::ppo::lr::KLScheduler`]) drives.
pub fn adamw_decayed_config(beta1: f64, beta2: f64, eps: f64, wd: f64) -> AdamWConfig {
    AdamWConfig::new()
        .with_beta_1(beta1 as f32)
        .with_beta_2(beta2 as f32)
        .with_epsilon(eps as f32)
        .with_weight_decay(wd as f32)
}

/// Build the **exempt** AdamW config (bias / norm / σ): identical to
/// [`adamw_decayed_config`] but with `weight_decay = 0`.
pub fn adamw_exempt_config(beta1: f64, beta2: f64, eps: f64) -> AdamWConfig {
    adamw_decayed_config(beta1, beta2, eps, 0.0)
}

/// A pair of AdamW optimisers (decayed + exempt) for one module `M`.
pub type AdamWOpt<M, B> = OptimizerAdaptor<AdamW, M, B>;

/// Initialise the two AdamW optimisers for module `M` from PPO hyper-parameters.
///
/// Returns `(decayed, exempt)`. Both share `beta1/beta2/eps`; only `decayed`
/// applies `wd`. Grad clipping is handled separately by
/// [`crate::ppo::clip::apply_global_norm_clip`] (global-norm, not burn's
/// per-tensor clip), so neither config sets `grad_clipping`.
pub fn init_adamw_pair<B, M>(
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
) -> (AdamWOpt<M, B>, AdamWOpt<M, B>)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let decayed = adamw_decayed_config(beta1, beta2, eps, wd).init::<B, M>();
    let exempt = adamw_exempt_config(beta1, beta2, eps).init::<B, M>();
    (decayed, exempt)
}

// ---------------------------------------------------------------------------
// Path-tracking exemption classifier.
// ---------------------------------------------------------------------------

/// Walks a module, tracking the field-name path stack via `enter_module` /
/// `exit_module`, and records the [`ParamId`]s whose path is
/// [`is_decay_exempt`]. Optionally collects `(path, exempt)` pairs for debugging
/// the actual param names (the documented migration risk).
struct ExemptVisitor {
    /// Current path stack of field names (e.g. `["policy_norm", "weight"]`).
    stack: Vec<String>,
    /// Param ids classified as decay-exempt.
    exempt: HashSet<ParamId>,
    /// Optional debug log of every visited `(joined_path, is_exempt)`.
    debug: Option<Vec<(String, bool)>>,
}

impl ExemptVisitor {
    fn joined(&self) -> String {
        self.stack.join(".")
    }
}

impl<B: Backend> ModuleVisitor<B> for ExemptVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let path = self.joined();
        let exempt = is_decay_exempt(&path);
        if let Some(log) = self.debug.as_mut() {
            log.push((path, exempt));
        }
        if exempt {
            self.exempt.insert(param.id);
        }
    }
}

/// Collect the set of [`ParamId`]s in `module` that are decay-exempt
/// (bias / norm / σ), keyed by their module path via [`is_decay_exempt`].
pub fn collect_decay_exempt_ids<B, M>(module: &M) -> HashSet<ParamId>
where
    B: Backend,
    M: Module<B>,
{
    let mut visitor = ExemptVisitor {
        stack: Vec::new(),
        exempt: HashSet::new(),
        debug: None,
    };
    module.visit(&mut visitor);
    visitor.exempt
}

/// Debug helper: return every parameter's `(module_path, is_exempt)` in visit
/// order. Use this to validate the name patterns against the real burn-derived
/// paths before trusting [`is_decay_exempt`] (spec migration risk).
pub fn debug_param_paths<B, M>(module: &M) -> Vec<(String, bool)>
where
    B: Backend,
    M: Module<B>,
{
    let mut visitor = ExemptVisitor {
        stack: Vec::new(),
        exempt: HashSet::new(),
        debug: Some(Vec::new()),
    };
    module.visit(&mut visitor);
    visitor.debug.unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Gradient splitting.
// ---------------------------------------------------------------------------

/// Moves gradients whose [`ParamId`] is in `exempt` out of the source container
/// into a fresh `exempt` container, leaving the non-exempt (decayed) gradients
/// behind. Rank-generic via a [`ModuleVisitor`] (the only way to touch the
/// type-erased [`GradientsParams`] at each param's static rank `D`), exactly as
/// [`crate::ppo::clip`] does.
struct SplitVisitor<'a, B: AutodiffBackend> {
    /// Source container; exempt grads are removed from here.
    src: &'a mut GradientsParams,
    /// Destination for the exempt grads.
    exempt_out: &'a mut GradientsParams,
    /// Ids to move into `exempt_out`.
    exempt_ids: &'a HashSet<ParamId>,
    _backend: core::marker::PhantomData<B>,
}

impl<B> ModuleVisitor<B> for SplitVisitor<'_, B>
where
    B: AutodiffBackend,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.exempt_ids.contains(&param.id) {
            if let Some(grad) = self.src.remove::<B::InnerBackend, D>(param.id) {
                self.exempt_out
                    .register::<B::InnerBackend, D>(param.id, grad);
            }
        }
    }
}

/// Split `grads` in place into `(decayed, exempt)`.
///
/// `grads` is mutated to retain ONLY the decayed (non-exempt) gradients and is
/// returned as the first element; the exempt gradients are moved into the second
/// element. `module` supplies the `(ParamId, rank)` enumeration (the container is
/// rank-opaque), and `exempt_ids` is the set from [`collect_decay_exempt_ids`].
pub fn split_grads_by_decay<B, M>(
    mut grads: GradientsParams,
    module: &M,
    exempt_ids: &HashSet<ParamId>,
) -> (GradientsParams, GradientsParams)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut exempt_out = GradientsParams::new();
    let mut visitor = SplitVisitor::<B> {
        src: &mut grads,
        exempt_out: &mut exempt_out,
        exempt_ids,
        _backend: core::marker::PhantomData,
    };
    module.visit(&mut visitor);
    (grads, exempt_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_is_exempt() {
        assert!(is_decay_exempt("policy_head_hold.bias"));
        assert!(is_decay_exempt("cross_attn.query.bias"));
    }

    #[test]
    fn norm_params_are_exempt() {
        // RMSNorm scale (weight) AND bias both sit under a *_norm path.
        assert!(is_decay_exempt("policy_norm.weight"));
        assert!(is_decay_exempt("policy_norm.bias"));
        assert!(is_decay_exempt("policy_perceiver.0.self_norm.weight"));
        assert!(is_decay_exempt("ffn_norm.weight"));
    }

    #[test]
    fn sigma_and_gamma_beta_exempt() {
        assert!(is_decay_exempt("noisy.weight_sigma"));
        assert!(is_decay_exempt("ln.gamma"));
        assert!(is_decay_exempt("ln.beta"));
    }

    #[test]
    fn matrix_weights_are_decayed() {
        assert!(!is_decay_exempt("policy_proj.weight"));
        assert!(!is_decay_exempt("value_head.weight"));
        assert!(!is_decay_exempt("cross_attn.query.weight"));
        assert!(!is_decay_exempt("ffn_gate.weight"));
        assert!(!is_decay_exempt("policy_head_fire.weight"));
    }
}
