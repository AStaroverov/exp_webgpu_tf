import * as tf from "@tensorflow/tfjs";
import type { Scalar } from "@tensorflow/tfjs-core/dist/tensor";
import type { NamedTensor } from "@tensorflow/tfjs-core/dist/tensor_types";
import { flatTypedArray } from "../utils/flat.ts";
import type { PreparedBatch } from "../memory/Memory.ts";
import {
  arrayHealthCheck,
  asyncUnwrapTensor,
  onReadyRead,
  syncUnwrapTensor,
} from "../utils/Tensor.ts";
import { normalize } from "../../../../lib/math.ts";

// Additive invalid-action mask sentinel: 0 = allowed, MASK_NEG = forbidden.
// Not -Infinity: an all-masked head would otherwise yield NaN after softmax/multinomial.
export const MASK_NEG = -1e9;

/**
 * Slice a flat per-step mask [B, sum(dims)] into per-head tensors [B, dim_i],
 * mirroring how the concatenated logits heads are arranged.
 */
export function splitMaskFlat(maskFlat: tf.Tensor, dims: number[]): tf.Tensor[] {
  return tf.tidy(() => {
    const heads: tf.Tensor[] = [];
    let offset = 0;
    for (const dim of dims) {
      heads.push(maskFlat.slice([0, offset], [-1, dim]));
      offset += dim;
    }
    return heads;
  });
}

export function trainPolicyNetwork(
  network: tf.LayersModel,
  states: tf.Tensor[],
  actions: tf.Tensor, // [B, actionDim]
  oldLogProbs: tf.Tensor, // [B]
  advantages: tf.Tensor, // [B]
  clipRatio: number,
  entropyCoeff: number,
  clipNorm: number,
  returnCost: boolean,
  masks?: tf.Tensor, // [B, sum(dims)] additive mask, optional
): { loss?: tf.Tensor; entropy: number } {
  let entropyValue = 0;
  const loss = tf.tidy(() => {
    return optimize(
      network.optimizer,
      () => {
        const predicted = network.apply(states, { training: true }) as tf.Tensor | tf.Tensor[];
        const logitsHeads = parsePolicyOutput(predicted);
        const maskHeads =
          masks != null
            ? splitMaskFlat(
                masks,
                logitsHeads.map((h) => h.shape[h.rank - 1] as number),
              )
            : undefined;

        // Compute log probabilities for categorical distributions
        const newLogProbs = computeLogProbCategorical(actions, logitsHeads, maskHeads); // [B]

        // r = exp(newLogProb - oldLogProb)
        const ratios = tf.exp(newLogProbs.sub(oldLogProbs)); // [B]

        // PPO clipped surrogate
        const clippedRatio = ratios.clipByValue(1 - clipRatio, 1 + clipRatio);
        const surr1 = ratios.mul(advantages);
        const surr2 = clippedRatio.mul(advantages);
        const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1) as tf.Scalar;

        // // SPO:  A * r  -  |A| * (r - 1)^2 / (2 * clipRatio)
        // const quad = ratios.sub(1).square().div(2 * clipRatio);                // [B]
        // const spoTerm = advantages.mul(ratios).sub(tf.abs(advantages).mul(quad)); // [B]
        // const policyLoss = spoTerm.mean().mul(-1) as tf.Scalar;              // scalar

        // Entropy for categorical distributions
        const rawEntropy = computeEntropyCategorical(logitsHeads, maskHeads);
        entropyValue = rawEntropy.dataSync()[0];

        // L2 anchor on raw (pre-mask) logits: softmax is shift-invariant and
        // saturated actions get ~zero entropy gradient, so without an anchor
        // the logits drift unboundedly. See PpoConfig.policyLogitsL2.
        // const logitsPenalty = logitsL2Coeff > 0
        //     ? logitsHeads
        //         .map(h => h.square().mean() as tf.Scalar)
        //         .reduce((acc, p) => acc.add(p) as tf.Scalar)
        //         .mul(logitsL2Coeff)
        //     : tf.scalar(0);

        // Total loss = *PO - α * H(π) + c * mean(logits²)
        const totalLoss = policyLoss.sub(rawEntropy.mul(entropyCoeff));
        return totalLoss as tf.Scalar;
      },
      { clipNorm, returnCost },
    );
  });
  return { loss, entropy: entropyValue };
}

export function trainValueNetwork(
  network: tf.LayersModel,
  states: tf.Tensor[],
  returns: tf.Tensor,
  oldValues: tf.Tensor,
  clipRatio: number,
  lossCoeff: number,
  clipNorm: number,
  returnCost: boolean,
): undefined | tf.Tensor {
  return tf.tidy(() => {
    return optimize(
      network.optimizer,
      () => {
        const newValues = (network.apply(states, { training: true }) as tf.Tensor).squeeze();
        const newValuesClipped = oldValues.add(
          newValues.sub(oldValues).clipByValue(-clipRatio, clipRatio),
        );
        const vfLoss1 = returns.sub(newValues).square();
        const vfLoss2 = returns.sub(newValuesClipped).square();

        const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(lossCoeff) as tf.Scalar;
        return finalValueLoss as tf.Scalar;
      },
      { clipNorm, returnCost },
    );
  });
}

export function computeKullbackLeiblerAprox(
  policyNetwork: tf.LayersModel,
  states: tf.Tensor[],
  actions: tf.Tensor,
  oldLogProb: tf.Tensor,
  batchSize: number,
  masks?: tf.Tensor, // [B, sum(dims)] additive mask, optional
) {
  return tf.tidy(() => {
    const predicted = policyNetwork.predict(states, { batchSize });
    const logitsHeads = parsePolicyOutput(predicted);
    const maskHeads =
      masks != null
        ? splitMaskFlat(
            masks,
            logitsHeads.map((h) => h.shape[h.rank - 1] as number),
          )
        : undefined;
    const newLogProbs = computeLogProbCategorical(actions, logitsHeads, maskHeads);
    // Schulman KL approximation: KL ≈ E[ratio - 1 - log(ratio)]
    const logRatio = newLogProbs.sub(oldLogProb);
    const ratio = logRatio.exp();
    const kl = ratio.sub(1).sub(logRatio).mean();
    return kl;
  });
}

export type BatchActResult = { actions: Float32Array; logits: Float32Array; logProb: number };

/**
 * Sample one action per batch row from the policy. WebGPU/async path: build the
 * whole sampling graph synchronously (`apply` → mask → sample → logProb) inside a
 * `tf.tidy` that KEEPS only the three small per-sample output tensors, then read
 * them back with the async `.data()` so the caller's thread is never blocked on
 * the GPU→CPU readback (a `.dataSync()` here would stall the queue — the opposite
 * of what we want). `tf.tidy` cannot wrap an `await`, hence the keep-then-read
 * split with a manual dispose. The passed-in `inputTensors` are owned by the
 * caller (created outside this tidy) and disposed there.
 */
export async function batchActAsync(
  policyNetwork: tf.LayersModel,
  inputTensors: tf.Tensor[],
  outputMasks?: Float32Array[], // one flat [sum(dims)] mask per state, optional
  options?: { greedy?: boolean },
): Promise<BatchActResult[]> {
  const batchSize = inputTensors[0].shape[0] as number;

  // 3 tensors per sample, flattened: [logitsFlat_0, actions_0, logProb_0, …].
  const kept = tf.tidy(() => {
    const predicted = policyNetwork.apply(inputTensors) as tf.Tensor | tf.Tensor[];
    const logitsHeads = parsePolicyOutput(predicted);
    const headDims = logitsHeads.map((h) => h.shape[h.rank - 1] as number);
    const out: tf.Tensor[] = [];

    for (let i = 0; i < batchSize; i++) {
      const stateLogitsHeads = logitsHeads.map((logits) => logits.slice([i], [1]));
      const flatMask = outputMasks?.[i];
      const maskHeads =
        flatMask != null
          ? splitMaskFlat(tf.tensor2d(flatMask, [1, flatMask.length]), headDims)
          : undefined;
      const squeezedMaskHeads = maskHeads?.map((h) => h.squeeze() as tf.Tensor);
      const sample = sampleActionsFromLogits(
        stateLogitsHeads.map((h) => h.squeeze()),
        squeezedMaskHeads,
        options,
      );
      const logProb = computeLogProbCategorical(
        sample.actions.expandDims(0),
        stateLogitsHeads,
        maskHeads,
      );
      out.push(sample.logitsFlat, sample.actions, logProb);
    }

    return out;
  });

  const datas = (await Promise.all(kept.map((t) => t.data()))) as Float32Array[];
  kept.forEach((t) => t.dispose());

  const results: BatchActResult[] = [];
  for (let i = 0; i < batchSize; i++) {
    const logits = datas[i * 3] as Float32Array;
    const actions = datas[i * 3 + 1] as Float32Array;
    const logProb = datas[i * 3 + 2][0];
    if (!arrayHealthCheck(logits) || !arrayHealthCheck(actions)) {
      throw new Error("Invalid tensor value");
    }
    results.push({ logits, actions, logProb });
  }
  return results;
}

export function sampleActionsFromLogits(
  heads: tf.Tensor[],
  maskHeads?: tf.Tensor[],
  opts?: { greedy?: boolean },
): { actions: tf.Tensor; logitsFlat: tf.Tensor } {
  const { greedy = false } = opts ?? {};

  return tf.tidy(() => {
    const actions = heads.map((logits, headIdx) => {
      const headMask = maskHeads?.[headIdx];
      // Mask before sampling/argMax so a forbidden action is never picked.
      const maskedLogits = headMask != null ? logits.add(headMask) : logits;
      return greedy
        ? tf.argMax(maskedLogits, -1)
        : tf.multinomial(maskedLogits as tf.Tensor1D, 1).squeeze();
    });

    return {
      actions: tf.stack(actions, -1),
      // Store RAW (unmasked) logits; mask is stored separately and re-applied at train time.
      logitsFlat: tf.concat(heads, -1),
    };
  });
}

function optimize(
  optimizer: tf.Optimizer,
  predict: () => Scalar,
  options?: { returnCost?: boolean; clipNorm?: number },
): undefined | tf.Scalar {
  const clipNorm = options?.clipNorm ?? 1;
  const returnCost = options?.returnCost ?? false;

  return tf.tidy(() => {
    const { grads, value } = tf.variableGrads(predict);

    // считаем общую норму градиентов
    const gradsArray = Object.values(grads).map((g) => g.square().sum());
    const sumSquares = gradsArray.reduce((acc, t) => acc.add(t), tf.scalar(0));
    const globalNorm = sumSquares.sqrt();
    // вычисляем множитель для клиппинга
    const eps = 1e-8;
    const safeGlobalNorm = tf.maximum(globalNorm, tf.scalar(eps));
    const clipCoef = tf.minimum(tf.scalar(1), tf.div(clipNorm, safeGlobalNorm));

    // применяем клиппинг к каждому градиенту
    const clippedGrads: NamedTensor[] = [];
    for (const [varName, grad] of Object.entries(grads)) {
      clippedGrads.push({ name: varName, tensor: tf.mul(grad, clipCoef) });
    }

    // fix for internal implementation of applyGradients
    clippedGrads.sort((a, b) => a.name.localeCompare(b.name));

    // применяем обрезанные градиенты
    optimizer.applyGradients(clippedGrads);

    tf.dispose(grads);

    return returnCost ? value : undefined;
  });
}

export function computeRetraceTargets<S>(
  policyNetwork: tf.LayersModel,
  valueNetwork: tf.LayersModel,
  createInputTensors: (batch: S[]) => tf.Tensor[],
  batch: PreparedBatch<S>,
  batchSize: number,
  gamma: number,
  actionDim: number,
  lambda: number = 0.95,
  masks?: Float32Array[], // one flat mask per step, optional
): {
  advantages: Float32Array;
  tdErrors: Float32Array;
  returns: Float32Array;
  values: Float32Array;
  pureLogits: Float32Array[];
} {
  return tf.tidy(() => {
    const input = createInputTensors(batch.states);
    const predicted = policyNetwork.predict(input, { batchSize });
    const logitsHeads = parsePolicyOutput(predicted);
    const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, actionDim]);
    const flatMasks = masks ?? batch.masks;
    const maskTensor =
      flatMasks != null && flatMasks.length > 0
        ? tf.tensor2d(flatTypedArray(flatMasks), [batch.size, flatMasks[0].length])
        : undefined;
    const maskHeads =
      maskTensor != null
        ? splitMaskFlat(
            maskTensor,
            logitsHeads.map((h) => h.shape[h.rank - 1] as number),
          )
        : undefined;
    const logProbCurrentTensor = computeLogProbCategorical(actions, logitsHeads, maskHeads);
    const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
    // Clamp the log-ratio BEFORE exp: ρ can overflow to Inf in f32 long before
    // min(1, ρ) ever sees it (e.g. a μ-suppressed action the current π likes).
    const rhosTensor = logProbCurrentTensor.sub(logProbBehaviorTensor).clipByValue(-20, 20).exp();
    const valuesTensor = (valueNetwork.predict(input, { batchSize }) as tf.Tensor).squeeze();

    const values = syncUnwrapTensor(valuesTensor) as Float32Array;
    const rhos = syncUnwrapTensor(rhosTensor) as Float32Array;

    const { advantages, tdErrors, retraceReturns } = computeRetrace(
      batch.rewards,
      batch.dones,
      values,
      rhos,
      gamma,
      lambda,
    );

    if (!arrayHealthCheck(advantages)) {
      throw new Error("Retrace advantages contain NaN");
    }
    if (!arrayHealthCheck(tdErrors)) {
      throw new Error("Retrace tdErrors contain NaN");
    }
    if (!arrayHealthCheck(retraceReturns)) {
      throw new Error("Retrace returns contain NaN");
    }

    const normalizedAdvantages = normalize(advantages);
    return {
      advantages: normalizedAdvantages as Float32Array,
      tdErrors: tdErrors,
      returns: retraceReturns,
      values: values,
      pureLogits: logitsHeads.map(syncUnwrapTensor) as Float32Array[],
    };
  });
}

/**
 * V-trace(λ) for state values — Espeholt et al., "IMPALA", 2018 (the state-value
 * specialization of Retrace, Munos et al. 2016). Reference: deepmind/scalable_agent
 * vtrace.py with ρ̄ = c̄ = 1 plus λ on the trace. See packages/ppo/PPO_RETRACE.md §3.
 *
 * ρ̂_t = min(1, ρ_t),  c_t = λ · ρ̂_t
 * δ_t  = ρ̂_t · (r_t + γ·V(s_{t+1}) − V(s_t))   ← leading IS weight on the TD term:
 *        r_t was earned by μ's action, so the V-form needs ρ̂ on δ (the Q-form
 *        doesn't because its δ conditions on a_t and uses E_π[Q]).
 * Δ_t  = δ_t + γ · c_t · Δ_{t+1}               (product convention ∏_{i=t}^{k−1} c_i)
 * v_t  = V(s_t) + Δ_t                          (value target)
 * A_t  = ρ̂_t · (r_t + γ·v_{t+1} − V(s_t))     (PG advantage — NEXT state's target)
 */
export function computeRetrace(
  rewards: Float32Array,
  dones: Float32Array,
  values: Float32Array,
  rhos: Float32Array,
  gamma: number,
  lambda: number,
): { retraceReturns: Float32Array; tdErrors: Float32Array; advantages: Float32Array } {
  const T = rewards.length;
  if (dones[T - 1] !== 1) {
    throw new Error("Retrace requires last state to be terminal");
  }

  const retraceReturns = new Float32Array(T);
  const tdErrors = new Float32Array(T);
  const advantages = new Float32Array(T);

  let delta = 0; // Δ_{t+1}

  for (let t = T - 1; t >= 0; --t) {
    const nextValue = dones[t] ? 0 : values[t + 1];
    const discount = dones[t] ? 0 : gamma;
    const value = values[t];

    const rhoClipped = Math.min(1, rhos[t]);
    const c = lambda * rhoClipped;

    // Raw TD error kept unweighted for diagnostics (charts).
    const tdError = rewards[t] + discount * nextValue - value;
    tdErrors[t] = tdError;

    // Δ_t = ρ̂_t·δ_t + γ · c_t · Δ_{t+1}
    delta = rhoClipped * tdError + discount * c * delta;

    // v_t = V(s_t) + Δ_t
    retraceReturns[t] = value + delta;

    // A_t = ρ̂_t · (r_t + γ·v_{t+1} − V(s_t)) — uses the NEXT state's value
    // target (retraceReturns[t+1], already computed by the reverse scan).
    const nextTarget = dones[t] ? 0 : retraceReturns[t + 1];
    advantages[t] = rhoClipped * (rewards[t] + discount * nextTarget - value);
  }

  return { retraceReturns, advantages, tdErrors };
}

function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[]): tf.Tensor[] {
  // Return array of per-head logits tensors (a single-output model predicts a
  // bare Tensor, not a one-element array).
  return Array.isArray(prediction) ? prediction : [prediction];
}

// Compute log probability for categorical distributions
function computeLogProbCategorical(
  actions: tf.Tensor,
  heads: tf.Tensor[],
  maskHeads?: tf.Tensor[],
): tf.Tensor {
  return tf.tidy(() => {
    const logProbs = heads.map((rawLogits, i) => {
      const logits = maskHeads?.[i] != null ? rawLogits.add(maskHeads[i]) : rawLogits;
      const actionIndices = actions.slice([0, i], [-1, 1]).squeeze([-1]); // [B]
      const logSoftmax = tf.logSoftmax(logits); // [B, numClasses]
      // Use one-hot encoding to select the correct log probability
      const numClasses = logSoftmax.shape[1] as number;
      const oneHot = tf.oneHot(actionIndices.toInt(), numClasses); // [B, numClasses]
      return tf.sum(logSoftmax.mul(oneHot), -1); // [B]
    });
    return tf.addN(logProbs) as tf.Tensor1D; // Sum log probs across heads -> [B]
  });
}

// Compute entropy for categorical distributions
function computeEntropyCategorical(logitsHeads: tf.Tensor[], maskHeads?: tf.Tensor[]): tf.Tensor {
  return tf.tidy(() => {
    const entropies = logitsHeads.map((rawLogits, i) => {
      const logits = maskHeads?.[i] != null ? rawLogits.add(maskHeads[i]) : rawLogits;
      const probs = tf.softmax(logits); // [B, numClasses]
      const logProbs = tf.logSoftmax(logits); // [B, numClasses]
      return probs.mul(logProbs).sum(-1).mul(-1); // -sum(p * log(p)) -> [B]
    });
    return tf.addN(entropies).div(logitsHeads.length).mean() as tf.Scalar; // Mean entropy across batch and heads
  });
}

export function networkHealthCheck<S>(
  network: tf.LayersModel,
  createInputTensors: (batch: S[]) => tf.Tensor[],
  prepareRandomInputArrays: () => S,
): Promise<boolean> {
  const tData = tf.tidy(() => {
    const inputs = createInputTensors([prepareRandomInputArrays()]);
    const result = network.predict(inputs) as tf.Tensor | tf.Tensor[];
    const arr = Array.isArray(result) ? result : [result];
    return arr.map((t) => t.squeeze());
  });

  return onReadyRead()
    .then(() => Promise.all(tData.map((t) => asyncUnwrapTensor(t))))
    .then(() => true);
}
