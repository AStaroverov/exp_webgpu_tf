# PPO + Retrace(λ) — how it should work

Reference document for the async actor-learner PPO with a Retrace(λ) critic used in
this repo (discrete actions, action masking, ε-exploration, browser/tfjs workers).
Everything below is verified against the primary sources (see References); the last
section lists where our implementation deviates.

---

## 1. Three policies, two ratios — the core invariant

In an async setup there are **three distinct distributions** per step, all defined
over the SAME masked action set:

| Symbol           | What it is                                                                                                                                   | Where it lives                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **μ** (behavior) | The distribution the action was ACTUALLY sampled from on the actor: stale network snapshot, **after** the action mask **and after** ε-mixing | Recorded at act time (`logProb`)         |
| **π_old**        | The learner's policy snapshot anchoring the PPO trust region (no ε)                                                                          | Snapshot at the start of an update phase |
| **π** (current)  | The learner's policy being optimized, recomputed every gradient step                                                                         | Learner                                  |

And **two ratios with different jobs** that must never be conflated:

```
ρ_t = π(a_t|s_t) / μ(a_t|s_t)        — off-policy correction (Retrace/V-trace, critic + advantage)
r_t = π(a_t|s_t) / π_old(a_t|s_t)    — trust region (PPO/SPO surrogate, actor)
```

- ρ corrects for _what actually generated the data_ (staleness + exploration).
- r bounds _how far the current update may move_ from a recent anchor.
- Using μ as the surrogate denominator merges the two; it is acceptable **only at
  low staleness and small ε** (OpenAI Five ran this way at staleness ≤ 1 version).
  Under high lag it anchors the clip to a stale policy and step sizes collapse.
- Applying ρ-corrected advantages AND a π/μ surrogate ratio double-corrects —
  exactly one off-policy correction belongs in the advantage.

## 2. Acting (data collection)

Per decision step the actor must:

1. Compute raw logits with the actor's network snapshot.
2. Apply the additive action mask: `logits' = logits + mask`, `mask ∈ {0, MASK_NEG}`.
   `MASK_NEG` is a large finite negative (−1e8…−1e9), **not** −Inf (0·(−Inf) = NaN).
3. Mix exploration **on the masked distribution**, then **re-mask**:
   `μ(a) ∝ (1−ε)·softmax(logits')_a + ε·u_a`, with the uniform/Dirichlet floor
   re-masked so forbidden actions stay at probability ~0. Sampling from a softmax
   of the re-masked mixed logits implicitly renormalizes over the valid set.
4. Sample `a ~ μ` and record:
   - the action index,
   - **log μ(a)** — the log-prob under the exact mixed, masked, renormalized
     distribution that was sampled from (NOT the clean π),
   - the mask itself (re-used verbatim at train time),
   - the raw (unmasked) logits if needed for diagnostics.

The recorded log μ is the single source of truth for the behavior policy. Computing
it from the clean π instead biases every ρ: an ε-explored, π-suppressed action
records log π ≈ −hundreds while its true behavior probability is ≥ ε/n — the
resulting `exp(log π_new − log μ)` explodes, poisoning traces, losses and KL.

## 3. Critic: Retrace(λ) targets for V

Retrace (Munos et al. 2016) builds a bootstrapped value target from TD errors
weighted by truncated importance traces. State-value form, per trajectory, with
`ρ_t = exp(logπ_current(a_t) − logμ(a_t))`:

```
c_t      = λ · min(1, ρ_t)                                   # trace coefficient
δ_t      = r_t + γ·(1−done_t)·V(s_{t+1}) − V(s_t)            # TD error (current critic)
Δ_t      = δ_t + γ·(1−done_t)·c_{t+1}·Δ_{t+1}                # reverse scan, Δ_T init from tail
v_ret_t  = V(s_t) + Δ_t                                      # value target
A_t      = Δ_t                                               # advantage for the actor
```

Key properties (verified against Munos 2016 / IMPALA):

- `min(1, ρ)` caps each trace step ⇒ the product is bounded ⇒ finite variance and
  contraction **regardless of how off-policy μ is** ("safe"). λ adds the usual
  eligibility-style bias/variance control.
- Retrace-for-V is V-trace with `c̄ = 1` and **no ρ̄-clip on δ** (equivalent to
  ρ̄ = ∞ ⇒ the target estimates V^π, unbiased fixed point). IMPALA instead clips
  the leading TD weight with `ρ̄ = 1` (slightly biased fixed point, lower variance)
  and the PG weight with a separate `clip_pg_rho_threshold`.
- On-policy (ρ = 1, λ = 1) it degenerates to n-step returns — a good sanity check.
- Do the reverse scan on CPU arrays, not per-step GPU tensor ops.
- Targets and advantages are constants for the update — stop-gradient / unwrap to
  plain arrays before the losses.
- IMPALA's PG advantage uses the **next state's target**:
  `A_t = ρ̂_t·(r_t + γ·v_ret_{t+1} − V(s_t))`. Our Δ-form is the Retrace analogue;
  using `v_ret_t` in place of `v_ret_{t+1}` there is a classic off-by-one bug.

## 4. Actor: PPO update

Canonical update over a collected batch:

```
for epoch in 1..K:                      # K = 4 (discrete canon)
    shuffle, split into minibatches
    for each minibatch:
        logits  = π(states)             # current params
        logits' = logits + stored_mask  # SAME mask as collection
        logp_new = logSoftmax(logits')[action]
        r = exp(logp_new − logp_anchor) # anchor: π_old (ideally), μ (low-staleness shortcut)
        A = normalize(advantages_mb)    # per minibatch, (A−mean)/(std+1e−8); never normalize returns
        L_pg   = −mean( min(r·A, clip(r, 1−ε_clip, 1+ε_clip)·A) )
        H      = −Σ where(valid, p·logp, 0)        # masked, NaN-safe entropy
        L_reg  = β·mean(logits²)                   # logit L2 anchor (see below)
        loss   = L_pg − c_ent·H + L_reg
        clip global grad norm; optimizer step      # step per MINIBATCH
    kl ≈ mean((exp(logratio)−1) − logratio)        # Schulman k3 estimator, non-negative
    if kl > 1.5·target_kl: break                   # early stop epochs
```

Value net: regress `V(s)` toward `v_ret` (MSE, optionally PPO2-style clipped around
the stored V with the same ε_clip; the "37 details" study found value clipping
often doesn't help). Coefficient 0.5 when sharing a trunk; separate nets can use
separate optimizers/LRs.

### Logit hygiene (discrete-specific)

- Softmax is shift-invariant ⇒ the all-ones direction is a null space of the loss;
  once an action saturates, its entropy gradient also vanishes ⇒ nothing stops
  monotone logit drift into the hundreds. Symptoms on a logit chart: frozen
  ranking, monotone divergence, eventual Inf/NaN.
- Remedy: **logit L2** `β·mean(logits²)`, β ~ 1e-4…1e-2 (AlphaStar-style), and/or
  KL-to-reference. Entropy bonus alone does NOT fix saturation (its gradient dies
  with the probabilities).
- Entropy under masks: zero the masked terms _before_ summation
  (`where(valid, p·logp, 0)`); max is `log(n_valid)`. Normalizing by `log(n_valid)`
  is optional and only matters when the valid-set size varies a lot (guard n=1).
- Masking is part of the distribution definition: the same stored mask must be
  applied at sampling, at logp recomputation, in entropy, and at eval. "Naive
  masking" (sample masked, score unmasked) gives biased gradients and KL explosion
  (Huang & Ontañón).

### Canonical hyperparameters (discrete)

| Knob                    | Canon                                                                            |
| ----------------------- | -------------------------------------------------------------------------------- |
| clip ε                  | 0.1–0.2                                                                          |
| epochs × minibatches    | 4 × 4                                                                            |
| advantage norm          | per minibatch                                                                    |
| entropy coef            | ~0.01 (or adaptive-α toward a target entropy — non-canonical add-on)             |
| target KL (early stop)  | 0.01–0.05, break at ~1.5×                                                        |
| grad clip (global norm) | 0.5                                                                              |
| Adam eps                | 1e-5 (not the framework default)                                                 |
| LR                      | 2.5e-4…3e-4, linearly annealed (or KL-adaptive ×0.95/×1.05 variant)              |
| γ / λ                   | 0.99 / 0.95                                                                      |
| staleness               | keep actors within ~1–2 versions (OpenAI Five); Sample Factory: <20–30 SGD steps |

## 5. Pitfall checklist

- [ ] log μ recorded from the EXACT sampling distribution (mask + ε mixing + renorm),
      from the acting forward pass (not recomputed in a different mode/backend).
- [ ] ε-mixing never resurrects masked actions (re-mask after mixing).
- [ ] Same per-step mask stored and re-applied at train time (logp, entropy, KL).
- [ ] Exactly one off-policy correction: ρ in the traces/advantage, r in the clip —
      not both, not swapped.
- [ ] Reverse-scan indexing: trace coefficient of the NEXT step multiplies the
      accumulated Δ; done_t zeroes both bootstrap and carry.
- [ ] Advantage uses the next-state value target (IMPALA form) — not raw V, not
      the same-step target.
- [ ] Targets/advantages detached from the graph.
- [ ] Advantages normalized (per minibatch); returns/value targets NOT normalized.
- [ ] Finite MASK_NEG; entropy zeroes masked terms before summation.
- [ ] Logit L2 (or KL-to-reference) present — entropy alone won't prevent saturation.
- [ ] KL estimator: `mean((r−1) − log r)`, not `mean(−logratio)`.
- [ ] Optimizer step per minibatch; global-norm grad clip (not per-value clip).
- [ ] clamp log-ratios (e.g. to [−20, 20]) before `exp` — ρ can overflow to Inf
      before `min(1, ·)` ever sees it.

## 6. Deviations in this codebase (audit list, 2026-06-04)

Mapping to `packages/ppo/src/core/train.ts` & learner:

1. **π_old ≡ μ**: `oldLogProbs` fed to the surrogate is the recorded behavior
   logprob. Since 2026-06-04 exploration is classic (no ε-mixing — the actor
   samples the masked softmax π itself, entropy does the exploring), so μ IS the
   actor-snapshot π and this is the canonical low-staleness PPO setup. Fine while
   actors stay within ~1 version (`versionDelta` chart). The loss is the standard
   PPO clip again (SPO variant kept commented out in `trainPolicyNetwork`).
2. ~~Retrace δ had no ρ̂ on the TD term and the advantage was `Δ_t`~~ — fixed
   2026-06-04: `computeRetrace` is now exact IMPALA V-trace (ρ̄ = c̄ = 1, + λ):
   `δ_t = ρ̂_t·(r+γV′−V)`, advantage `ρ̂_t·(r+γ·v_{t+1}−V)` from the next-state
   target. Note the SPO ratio (π/μ) multiplies this ρ̂-weighted advantage — the
   standard hybrid recipe accepts this overlap (the clip/quad bounds it).
3. **ε-uniform floor uses n_total (13), not n_valid** — harmless: μ is recorded
   from the re-masked, renormalized mixture (self-consistent), just not the
   textbook mixture.
4. **Logit L2** added as `policyLogitsL2` (1e-3) over raw logits of all heads —
   includes masked slots (they get pulled to 0; harmless, slightly stronger anchor).
5. ~~Adaptive entropy α (SAC-style)~~ — replaced 2026-06-04 with the canonical
   fixed `entropyCoeff: 0.01`.
6. **KL-adaptive LR** (×0.95/×1.05 between bounds) instead of linear anneal; KL
   estimator is the correct k3 form.
7. ~~Adam eps unset (tf default)~~ — fixed 2026-06-04: `AdamW` defaults to 1e-5.
8. ~~No clamp on the log-ratio before `exp`~~ — fixed 2026-06-04: clipped to
   [−20, 20] in `computeRetraceTargets`.
9. **Advantage normalization is per whole batch** (in `computeRetraceTargets`),
   not per minibatch — an accepted variant (SB3 does the same); left as is.
10. **clipNorm = 5** vs canonical 0.5 — interacts with the KL-adaptive LR; left
    as is, revisit if gradients look spiky.

## References

- Schulman et al. 2017, _PPO_ — arXiv:1707.06347
- Munos et al. 2016, _Safe and Efficient Off-Policy RL (Retrace λ)_ — arXiv:1606.02647
- Espeholt et al. 2018, _IMPALA (V-trace)_ — arXiv:1802.01561; reference `vtrace.py` in deepmind/scalable_agent
- Huang & Ontañón 2020, _A Closer Look at Invalid Action Masking_ — arXiv:2006.14171
- Huang et al., _The 37 Implementation Details of PPO_ — ICLR blog track 2022
- Schulman, _Approximating KL Divergence_ — joschu.net/blog/kl-approx
- Luo et al. 2019, _IMPACT_ (PPO+V-trace hybrid, target-net trust region) — arXiv:1912.00167
- OpenAI Five (staleness & sample reuse ablations) — cdn.openai.com/dota-2.pdf
