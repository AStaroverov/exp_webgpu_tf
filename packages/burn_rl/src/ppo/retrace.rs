//! Retrace(λ) advantage / return estimation (CPU reverse scan).
//!
//! Port of `computeRetrace` in `packages/ppo/src/core/train.ts`. Operates on
//! rollout buffers already read back from the GPU; plain scalar `f32` code with
//! no backend dependency (mirrors `gae.rs`).
//!
//! The input `log_ratios` are the raw log importance weights `log(π_new / μ)`.
//! They are clamped to `[-20, 20]` here before exponentiation to keep the
//! importance ratios `ρ = exp(logρ)` finite, then clipped via `ρ̂ = min(1, ρ)`.
//!
//! Reverse recursion (t: T→0):
//!
//! ```text
//! ρ̂   = min(1, exp(clamp(logρ, -20, 20)))
//! c   = λ·ρ̂
//! δ   = r + γ(1-d)·v[t+1] - v                 (raw TD error, unweighted)
//! Δ   = ρ̂·δ + γ(1-d)·c·Δ[t+1]
//! vtgt = v + Δ
//! A   = ρ̂·(r + γ(1-d)·vtgt[t+1] - v)
//! ```
//!
//! `dones[t]` (1.0 = terminal) zeroes both the bootstrap value and the discount
//! for that step, and resets the trace continuation, exactly as in the TS scan.

/// Lower / upper clamp applied to log importance ratios before exponentiation.
pub const LOG_RATIO_CLAMP_MIN: f32 = -20.0;
pub const LOG_RATIO_CLAMP_MAX: f32 = 20.0;

/// Output of the Retrace scan. All vectors have length `rewards.len()`.
#[derive(Clone, Debug, Default)]
pub struct RetraceOutput {
    /// Retrace value targets `v_t = V(s_t) + Δ_t` (DETACHED — plain data, no graph).
    pub returns: Vec<f32>,
    /// Retrace advantages `A_t = ρ̂_t·(r_t + γ(1-d)·v_tgt[t+1] - V(s_t))`.
    pub advantages: Vec<f32>,
    /// Raw, unweighted TD errors `r_t + γ(1-d)·v[t+1] - V(s_t)` (diagnostics).
    pub td_errors: Vec<f32>,
}

/// Compute Retrace(λ) returns / advantages / TD-errors over a flat rollout that
/// may span several episodes.
///
/// * `rewards`, `dones`, `values`, `log_ratios` are all length `T`. `values`
///   holds the on-policy value estimate `V(s_t)` for each step; the bootstrap
///   for the next state is read from `values[t + 1]` (and is forced to `0` on a
///   terminal step, so no separate `last_value` is required).
/// * `dones[t] == true` marks the terminal step of an episode.
/// * `log_ratios[t] = log(π_new(a_t|s_t) / μ(a_t|s_t))`, clamped here.
///
/// Matches the reference TS scan, which requires the final state to be terminal;
/// this implementation does not panic on that (callers should ensure it) but the
/// recursion is well-defined regardless because `values[t + 1]` is only read when
/// `dones[t]` is false and `t < T - 1`.
pub fn compute_retrace(
    rewards: &[f32],
    dones: &[bool],
    values: &[f32],
    log_ratios: &[f32],
    gamma: f32,
    lambda: f32,
) -> RetraceOutput {
    let t_len = rewards.len();
    debug_assert_eq!(dones.len(), t_len);
    debug_assert_eq!(values.len(), t_len);
    debug_assert_eq!(log_ratios.len(), t_len);

    let mut returns = vec![0.0f32; t_len];
    let mut advantages = vec![0.0f32; t_len];
    let mut td_errors = vec![0.0f32; t_len];

    if t_len == 0 {
        return RetraceOutput {
            returns,
            advantages,
            td_errors,
        };
    }

    // Δ_{t+1}, carried backwards through the scan.
    let mut delta_acc = 0.0f32;

    for t in (0..t_len).rev() {
        let nonterminal = if dones[t] { 0.0 } else { 1.0 };
        // On a terminal step there is no bootstrap and no discount continuation.
        let discount = gamma * nonterminal;
        let value = values[t];
        let next_value = if dones[t] || t == t_len - 1 {
            0.0
        } else {
            values[t + 1]
        };

        // ρ̂ = min(1, exp(clamp(logρ, -20, 20)))
        let clamped = log_ratios[t].clamp(LOG_RATIO_CLAMP_MIN, LOG_RATIO_CLAMP_MAX);
        let rho = clamped.exp();
        let rho_clipped = rho.min(1.0);
        let c = lambda * rho_clipped;

        // Raw, unweighted TD error (kept for diagnostics / charts).
        let td_error = rewards[t] + discount * next_value - value;
        td_errors[t] = td_error;

        // Δ_t = ρ̂_t·δ_t + γ(1-d)·c_t·Δ_{t+1}
        delta_acc = rho_clipped * td_error + discount * c * delta_acc;

        // v_t = V(s_t) + Δ_t
        returns[t] = value + delta_acc;

        // A_t = ρ̂_t·(r_t + γ(1-d)·v_tgt[t+1] − V(s_t)) — uses the NEXT step's
        // retrace target (already computed by the reverse scan).
        let next_target = if dones[t] || t == t_len - 1 {
            0.0
        } else {
            returns[t + 1]
        };
        advantages[t] = rho_clipped * (rewards[t] + discount * next_target - value);
    }

    RetraceOutput {
        returns,
        advantages,
        td_errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single terminal step: no bootstrap, ρ̂ = min(1, exp(0)) = 1.
    /// Δ = r - v, return = v + Δ = r, td = r - v, A = ρ̂·(r - v).
    #[test]
    fn single_terminal_step() {
        let out = compute_retrace(&[2.0], &[true], &[0.5], &[0.0], 0.99, 0.95);
        assert!((out.returns[0] - 2.0).abs() < 1e-6);
        assert!((out.td_errors[0] - 1.5).abs() < 1e-6);
        assert!((out.advantages[0] - 1.5).abs() < 1e-6);
    }

    /// Reproduce the TS reverse scan exactly for a two-step episode with a
    /// non-trivial importance ratio on the first step.
    #[test]
    fn matches_ts_reference_scan() {
        let rewards = [1.0f32, 3.0];
        let dones = [false, true];
        let values = [0.5f32, 1.0];
        // logρ on step 0 → ρ = exp(-0.5) ≈ 0.6065 (clipped < 1); step 1 → ρ = 1.
        let log_ratios = [-0.5f32, 0.0];
        let gamma = 0.99;
        let lambda = 0.95;

        let out = compute_retrace(&rewards, &dones, &values, &log_ratios, gamma, lambda);

        // Mirror the TS scan as the ground truth.
        let t = 2usize;
        let mut ret = vec![0.0f32; t];
        let mut adv = vec![0.0f32; t];
        let mut td = vec![0.0f32; t];
        let mut delta = 0.0f32;
        for i in (0..t).rev() {
            let nonterm = if dones[i] { 0.0 } else { 1.0 };
            let disc = gamma * nonterm;
            let nv = if dones[i] || i == t - 1 { 0.0 } else { values[i + 1] };
            let rho = log_ratios[i].clamp(-20.0, 20.0).exp().min(1.0);
            let c = lambda * rho;
            let tderr = rewards[i] + disc * nv - values[i];
            td[i] = tderr;
            delta = rho * tderr + disc * c * delta;
            ret[i] = values[i] + delta;
            let nt = if dones[i] || i == t - 1 { 0.0 } else { ret[i + 1] };
            adv[i] = rho * (rewards[i] + disc * nt - values[i]);
        }

        for i in 0..t {
            assert!((out.returns[i] - ret[i]).abs() < 1e-6);
            assert!((out.advantages[i] - adv[i]).abs() < 1e-6);
            assert!((out.td_errors[i] - td[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn log_ratio_is_clamped_before_exp() {
        // Huge logρ must not produce inf; ρ̂ = min(1, exp(20)) = 1.
        let out = compute_retrace(&[1.0], &[true], &[0.0], &[1000.0], 0.99, 0.95);
        assert!(out.returns[0].is_finite());
        assert!((out.advantages[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_rollout() {
        let out = compute_retrace(&[], &[], &[], &[], 0.99, 0.95);
        assert!(out.returns.is_empty());
        assert!(out.advantages.is_empty());
        assert!(out.td_errors.is_empty());
    }
}
