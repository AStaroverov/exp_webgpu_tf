import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import { flatTypedArray } from '../utils/flat.ts';
import { PreparedBatch } from '../memory/Memory.ts';
import { arrayHealthCheck, asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../utils/Tensor.ts';
import { shouldNoiseLayer } from '../models/noiseGate.ts';
import { normalize } from '../../../../lib/math.ts';

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
    actions: tf.Tensor,      // [B, actionDim]
    oldLogProbs: tf.Tensor,  // [B]
    advantages: tf.Tensor,   // [B]
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
    returnCost: boolean,
    masks?: tf.Tensor,       // [B, sum(dims)] additive mask, optional
): { loss?: tf.Tensor, entropy: number } {
    let entropyValue = 0;
    const loss = tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predicted = network.apply(states, { training: true, noise: shouldNoiseLayer }) as tf.Tensor | tf.Tensor[];
            const logitsHeads = parsePolicyOutput(predicted);
            const maskHeads = masks != null ? splitMaskFlat(masks, logitsHeads.map(h => h.shape[h.rank - 1] as number)) : undefined;

            // Compute log probabilities for categorical distributions
            const newLogProbs = computeLogProbCategorical(actions, logitsHeads, maskHeads); // [B]

            // r = exp(newLogProb - oldLogProb)
            const ratios = tf.exp(newLogProbs.sub(oldLogProbs));                 // [B]

            // // PPO
            // const clippedRatio = ratios.clipByValue(1 - clipRatio, 1 + clipRatio);
            // const surr1 = ratios.mul(advantages);
            // const surr2 = clippedRatio.mul(advantages);
            // const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

            // SPO:  A * r  -  |A| * (r - 1)^2 / (2 * clipRatio)
            const quad = ratios.sub(1).square().div(2 * clipRatio);                // [B]
            const spoTerm = advantages.mul(ratios).sub(tf.abs(advantages).mul(quad)); // [B]
            const policyLoss = spoTerm.mean().mul(-1) as tf.Scalar;              // scalar

            // Entropy for categorical distributions
            const rawEntropy = computeEntropyCategorical(logitsHeads, maskHeads);
            entropyValue = rawEntropy.dataSync()[0];

            // Total loss = SPO - α * H(π)
            const totalLoss = policyLoss.sub(rawEntropy.mul(entropyCoeff));
            return totalLoss as tf.Scalar;
        }, {clipNorm, returnCost});
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
        return optimize(network.optimizer, () => {
            const newValues = (network.apply(states, { training: true }) as tf.Tensor).squeeze();
            const newValuesClipped = oldValues.add(
                newValues.sub(oldValues).clipByValue(-clipRatio, clipRatio),
            );
            const vfLoss1 = returns.sub(newValues).square();
            const vfLoss2 = returns.sub(newValuesClipped).square();

            const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(lossCoeff) as tf.Scalar;
            return finalValueLoss as tf.Scalar;
        }, {clipNorm, returnCost});
    });
}

export function computeKullbackLeiblerAprox(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,
    oldLogProb: tf.Tensor,
    batchSize: number,
    masks?: tf.Tensor,       // [B, sum(dims)] additive mask, optional
) {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, {batchSize});
        const logitsHeads = parsePolicyOutput(predicted);
        const maskHeads = masks != null ? splitMaskFlat(masks, logitsHeads.map(h => h.shape[h.rank - 1] as number)) : undefined;
        const newLogProbs = computeLogProbCategorical(actions, logitsHeads, maskHeads);
        // Schulman KL approximation: KL ≈ E[ratio - 1 - log(ratio)]
        const logRatio = newLogProbs.sub(oldLogProb);
        const ratio = logRatio.exp();
        const kl = ratio.sub(1).sub(logRatio).mean();
        return kl;
    });
}

export function batchAct(
    policyNetwork: tf.LayersModel,
    inputTensors: tf.Tensor[],
    outputMasks?: Float32Array[],  // one flat [sum(dims)] mask per state, optional
    options?: { greedy?: boolean; epsilon?: number; noises?: (tf.Tensor[] | undefined)[] },
): {
    actions: Float32Array,
    logits: Float32Array,
    logProb: number,
}[] {
    return tf.tidy(() => {
        const batchSize = inputTensors[0].shape[0] as number;
        const noise = options?.greedy !== true ? shouldNoiseLayer : undefined;
        const predicted = policyNetwork.apply(inputTensors, { noise }) as tf.Tensor | tf.Tensor[];
        // batchAct fully consumes the caller-provided input tensors (forward pass only,
        // returns plain Float32Arrays), so dispose them here — no caller-side tidy needed.
        tf.dispose(inputTensors);
        const logitsHeads = parsePolicyOutput(predicted);
        const headDims = logitsHeads.map(h => h.shape[h.rank - 1] as number);

        const results: {actions: Float32Array, logits: Float32Array, logProb: number}[] = [];

        for (let i = 0; i < batchSize; i++) {
            const stateLogitsHeads = logitsHeads.map(logits => logits.slice([i], [1]));
            const noiseTensors = options?.noises?.[i];
            const flatMask = outputMasks?.[i];
            const maskHeads = flatMask != null
                ? splitMaskFlat(tf.tensor2d(flatMask, [1, flatMask.length]), headDims)
                : undefined;
            const squeezedMaskHeads = maskHeads?.map(h => h.squeeze() as tf.Tensor);
            const sample = sampleActionsFromLogits(stateLogitsHeads.map(h => h.squeeze()), noiseTensors, squeezedMaskHeads, options);
            const logProb = computeLogProbCategorical(sample.actions.expandDims(0), stateLogitsHeads, maskHeads);

            results.push({
                logits: syncUnwrapTensor(sample.logitsFlat) as Float32Array,
                actions: syncUnwrapTensor(sample.actions) as Float32Array,
                logProb: syncUnwrapTensor(logProb)[0],
            });
        }

        return results;
    });
}

export function sampleActionsFromLogits(
    heads: tf.Tensor[],
    noises?: tf.Tensor[],
    maskHeads?: tf.Tensor[],
    opts?: { greedy?: boolean; epsilon?: number },
): { actions: tf.Tensor; logitsFlat: tf.Tensor } {
    const { greedy = false, epsilon = 0 } = opts ?? {};

    return tf.tidy(() => {
        const results = heads.map((logits, headIdx) => {
            const headMask = maskHeads?.[headIdx];
            if (greedy) {
                // Apply mask before greedy argMax so eval never picks a forbidden action.
                const maskedLogits = headMask != null ? logits.add(headMask) : logits;
                return { action: tf.argMax(maskedLogits, -1), noisyLogits: maskedLogits };
            }
            return sampleCategorical(logits, headMask, epsilon, noises?.[headIdx]);
        });

        return {
            actions: tf.stack(results.map(r => r.action), -1),
            // Store RAW (unmasked) logits; mask is stored separately and re-applied at train time.
            logitsFlat: tf.concat(heads, -1),
        };
    });
}

function sampleCategorical(
    logits: tf.Tensor,
    mask: undefined | tf.Tensor,
    epsilon: number,
    dirichletNoise: undefined | tf.Tensor,
): { action: tf.Tensor } {
    return tf.tidy(() => {
        // Apply mask FIRST so ε-uniform / Dirichlet mixing happens on already-masked
        // logits and exploration never resurrects a forbidden action.
        if (mask != null) logits = logits.add(mask);
        const numActions = logits.shape[logits.rank - 1]!;
        let noisyLogits = logits;

        if (dirichletNoise != null && epsilon > 0) {
            // AlphaZero-style Dirichlet noise mixing:
            // p' = (1 - ε) * p + ε * Dir(α)
            const probs = tf.softmax(logits);
            const noisyProbs = probs.mul(1 - epsilon).add(dirichletNoise.mul(epsilon));
            noisyLogits = tf.log(noisyProbs.clipByValue(1e-8, 1));
        } else if (epsilon > 0) {
            const uniform = tf.fill(logits.shape, 1 / numActions);
            const probs = tf.softmax(logits).mul(1 - epsilon).add(uniform.mul(epsilon));
            noisyLogits = tf.log(probs.clipByValue(1e-8, 1));
        }

        // Re-apply the mask AFTER ε-uniform / Dirichlet mixing: the mixing re-adds a
        // nonzero floor (ε/numActions, or ε·Dir) to every slot, including forbidden
        // ones, so masking only before the mix would let multinomial resurrect an
        // invalid action. Re-masking here drives those slots back to MASK_NEG.
        if (mask != null) noisyLogits = noisyLogits.add(mask);

        const action = tf.multinomial(noisyLogits as tf.Tensor1D, 1).squeeze();
        return { action };
    });
}

export function pureAct<S>(
    policyNetwork: tf.LayersModel,
    createInputTensors: (batch: S[]) => tf.Tensor[],
    state: S,
    mask?: Float32Array,     // flat [sum(dims)] mask, optional
): {
    actions: Float32Array,
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.apply(createInputTensors([state])) as tf.Tensor | tf.Tensor[];
        const logitsHeads = parsePolicyOutput(predicted);
        const maskHeads = mask != null
            ? splitMaskFlat(tf.tensor2d(mask, [1, mask.length]), logitsHeads.map(h => h.shape[h.rank - 1] as number))
            : undefined;
        const actions = tf
            .concat(logitsHeads.map((logits, i) => {
                const masked = maskHeads?.[i] != null ? logits.add(maskHeads[i]) : logits;
                return tf.argMax(masked, -1).expandDims(-1);
            }), -1)
            .squeeze([0]);

        return {
            actions: syncUnwrapTensor(actions) as Float32Array,
        };
    });
}

function optimize(
    optimizer: tf.Optimizer,
    predict: () => Scalar,
    options?: { returnCost?: boolean, clipNorm?: number },
): undefined | tf.Scalar {
    const clipNorm = options?.clipNorm ?? 1;
    const returnCost = options?.returnCost ?? false;

    return tf.tidy(() => {
        const {grads, value} = tf.variableGrads(predict);

        // считаем общую норму градиентов
        const gradsArray = Object.values(grads).map(g => g.square().sum());
        const sumSquares = gradsArray.reduce((acc, t) => acc.add(t), tf.scalar(0));
        const globalNorm = sumSquares.sqrt();
        // вычисляем множитель для клиппинга
        const eps = 1e-8;
        const safeGlobalNorm = tf.maximum(globalNorm, tf.scalar(eps));
        const clipCoef = tf.minimum(tf.scalar(1), tf.div(clipNorm, safeGlobalNorm));

        // применяем клиппинг к каждому градиенту
        const clippedGrads: NamedTensor[] = [];
        for (const [varName, grad] of Object.entries(grads)) {
            clippedGrads.push({name: varName, tensor: tf.mul(grad, clipCoef)});
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
    masks?: Float32Array[],  // one flat mask per step, optional
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
    pureLogits: Float32Array[],
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const predicted = policyNetwork.predict(input, {batchSize});
        const logitsHeads = parsePolicyOutput(predicted);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, actionDim]);
        const flatMasks = masks ?? batch.masks;
        const maskTensor = flatMasks != null && flatMasks.length > 0
            ? tf.tensor2d(flatTypedArray(flatMasks), [batch.size, flatMasks[0].length])
            : undefined;
        const maskHeads = maskTensor != null
            ? splitMaskFlat(maskTensor, logitsHeads.map(h => h.shape[h.rank - 1] as number))
            : undefined;
        const logProbCurrentTensor = computeLogProbCategorical(actions, logitsHeads, maskHeads);
        const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
        const rhosTensor = logProbCurrentTensor.sub(logProbBehaviorTensor).exp();
        const valuesTensor = (valueNetwork.predict(input, {batchSize}) as tf.Tensor).squeeze();

        const values = syncUnwrapTensor(valuesTensor) as Float32Array;
        const rhos = syncUnwrapTensor(rhosTensor) as Float32Array;

        const {advantages, tdErrors, retraceReturns} = computeRetrace(
            batch.rewards,
            batch.dones,
            values,
            rhos,
            gamma,
            lambda,
        );

        if (!arrayHealthCheck(advantages)) {
            throw new Error('Retrace advantages contain NaN');
        }
        if (!arrayHealthCheck(tdErrors)) {
            throw new Error('Retrace tdErrors contain NaN');
        }
        if (!arrayHealthCheck(retraceReturns)) {
            throw new Error('Retrace returns contain NaN');
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
 * Retrace(λ) — Munos et al., "Safe and Efficient Off-Policy RL", 2016
 *
 * Trace coefficient: c_t = λ · min(1, ρ_t)
 * Target:  v^ret_t = V(s_t) + Δ_t
 *   where  Δ_t = δ_t + γ · c_t · Δ_{t+1}
 *          δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
 *
 * No separate ρ̄ on δ_t — off-policy correction is entirely in the trace coefficients.
 */
export function computeRetrace(
    rewards: Float32Array,
    dones: Float32Array,
    values: Float32Array,
    rhos: Float32Array,
    gamma: number,
    lambda: number,
): { retraceReturns: Float32Array, tdErrors: Float32Array, advantages: Float32Array } {
    const T = rewards.length;
    if (dones[T - 1] !== 1) {
        throw new Error('Retrace requires last state to be terminal');
    }

    const retraceReturns = new Float32Array(T);
    const tdErrors = new Float32Array(T);
    const advantages = new Float32Array(T);

    let delta = 0; // Δ_{t+1}

    for (let t = T - 1; t >= 0; --t) {
        const nextValue = dones[t] ? 0 : values[t + 1];
        const discount = dones[t] ? 0 : gamma;
        const value = values[t];

        // c_t = λ · min(1, ρ_t)
        const c = lambda * Math.min(1, rhos[t]);

        // δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        const tdError = rewards[t] + discount * nextValue - value;
        tdErrors[t] = tdError;

        // Δ_t = δ_t + γ · c_t · Δ_{t+1}
        delta = tdError + discount * c * delta;
        advantages[t] = delta;

        // v^ret_t = V(s_t) + Δ_t
        retraceReturns[t] = value + delta;
    }

    return {retraceReturns, advantages, tdErrors};
}

function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[]): tf.Tensor[] {
    // Return array of logits tensors: [shootLogits, moveLogits, rotLogits, turRotLogits]
    return (prediction as tf.Tensor[]);
}

// Compute log probability for categorical distributions
function computeLogProbCategorical(actions: tf.Tensor, heads: tf.Tensor[], maskHeads?: tf.Tensor[]): tf.Tensor {
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
        return arr.map(t => t.squeeze());
    });

    return onReadyRead()
        .then(() => Promise.all(tData.map(t => asyncUnwrapTensor(t))))
        .then(() => true);
}
