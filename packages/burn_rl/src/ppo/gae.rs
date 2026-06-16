//! CPU-side Generalised Advantage Estimation (reverse scan). Operates on the
//! rollout buffers already read back from the GPU; everything here is plain
//! scalar code with no backend dependency.

/// Generalised Advantage Estimation over a flat rollout that may span several
/// episodes. `dones[t]` marks the last step of an episode; `last_value` is the
/// bootstrap value of the state following the final step.
///
/// Returns `(advantages, returns)`, each of length `rewards.len()`.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut adv = vec![0.0f32; n];
    let mut last_gae = 0.0f32;
    for t in (0..n).rev() {
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };
        // On a terminal step there is no bootstrap and the GAE recursion resets.
        let nonterminal = if dones[t] { 0.0 } else { 1.0 };
        let delta = rewards[t] + gamma * next_value * nonterminal - values[t];
        last_gae = delta + gamma * lambda * nonterminal * last_gae;
        adv[t] = last_gae;
    }
    let returns: Vec<f32> = adv.iter().zip(values).map(|(a, v)| a + v).collect();
    (adv, returns)
}

/// Normalise advantages to zero mean / unit variance (standard PPO trick).
pub fn normalize(adv: &mut [f32]) {
    let n = adv.len() as f32;
    if n == 0.0 {
        return;
    }
    let mean = adv.iter().sum::<f32>() / n;
    let var = adv.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n;
    let std = var.sqrt() + 1e-8;
    for a in adv.iter_mut() {
        *a = (*a - mean) / std;
    }
}
