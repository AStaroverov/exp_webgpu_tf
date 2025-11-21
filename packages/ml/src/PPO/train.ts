import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import { normalize } from '../../../../lib/math.ts';
import { random } from '../../../../lib/random.ts';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import { flatTypedArray } from '../../../ml-common/flat.ts';
import { InputArrays, prepareRandomInputArrays } from '../../../ml-common/InputArrays.ts';
import { createInputTensors } from '../../../ml-common/InputTensors.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';
import { arrayHealthCheck, asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../../../ml-common/Tensor.ts';
import { ACTION_HEAD_DIMS } from '../Models/Create.ts';

// Removed: const C = 0.5 * Math.log(2 * Math.PI * Math.E);
// export function trainPolicyNetwork(
//     network: tf.LayersModel,
//     states: tf.Tensor[],
//     actions: tf.Tensor,      // [batchSize, actionDim]
//     oldLogProbs: tf.Tensor,  // [batchSize]
//     advantages: tf.Tensor,   // [batchSize]
//     batchSize: number,
//     clipRatio: number,
//     entropyCoeff: number,
//     clipNorm: number,
//     minLogStd: number[],
//     maxLogStd: number[],
//     returnCost: boolean,
// ): undefined | tf.Tensor {
//     return tf.tidy(() => {
//         return optimize(network.optimizer, () => {
//             const predicted = network.predict(states, { batchSize });
//             const { mean, logStd } = parsePolicyOutput(predicted, minLogStd, maxLogStd);
//             const newLogProbs = computeLogProbTanh(actions, mean, logStd.exp());
//             const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
//             const clippedRatio = ratio.clipByValue(1 - clipRatio, 1 + clipRatio);
//             const surr1 = ratio.mul(advantages);
//             const surr2 = clippedRatio.mul(advantages);
//             const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);
//             const entropy = logStd.add(C).sum(1).mean().mul(entropyCoeff);
//             const totalLoss = policyLoss.sub(entropy);
//             return totalLoss as tf.Scalar;
//         }, { clipNorm, returnCost });
//     });
// }
export function trainPolicyNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,      // [B, actionDim]
    oldLogProbs: tf.Tensor,  // [B]
    advantages: tf.Tensor,   // [B]
    batchSize: number,
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
    returnCost: boolean,
): undefined | tf.Tensor {
    return tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predicted = network.predict(states, {batchSize});
            const logitsHeads = parsePolicyOutput(predicted);

            // Compute log probabilities for categorical distributions
            const newLogProbs = computeLogProbCategorical(actions, logitsHeads); // [B]

            // r = exp(newLogProb - oldLogProb)
            const ratios = tf.exp(newLogProbs.sub(oldLogProbs));                 // [B]

            // SPO:  A * r  -  |A| * (r - 1)^2 / (2*clipRatio)
            const epsT = tf.scalar(clipRatio, 'float32');
            const quad = ratios.sub(1).square().div(epsT.mul(2));                // [B]
            const spoTerm = advantages.mul(ratios).sub(tf.abs(advantages).mul(quad)); // [B]
            const policyLoss = spoTerm.mean().mul(-1) as tf.Scalar;              // scalar

            // Entropy for categorical distributions
            const entropy = computeEntropyCategorical(logitsHeads).mul(entropyCoeff) as tf.Scalar;

            // Total loss
            const totalLoss = policyLoss.sub(entropy);
            return totalLoss as tf.Scalar;
        }, {clipNorm, returnCost});
    });
}

export function trainValueNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    returns: tf.Tensor,
    oldValues: tf.Tensor,
    batchSize: number,
    clipRatio: number,
    lossCoeff: number,
    clipNorm: number,
    returnCost: boolean,
): undefined | tf.Tensor {
    return tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const newValues = (network.predict(states, {batchSize}) as tf.Tensor).squeeze();
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
) {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, {batchSize});
        const logitsHeads = parsePolicyOutput(predicted);
        const newLogProbs = computeLogProbCategorical(actions, logitsHeads);
        const diff = oldLogProb.sub(newLogProbs).div(tf.scalar(ACTION_HEAD_DIMS.length));
        const kl = diff.mean().abs();
        return kl;
    });
}

export function noisyAct(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
    noise?: tf.Tensor[],
): {
    actions: Float32Array,
    logits: Float32Array,
    logProb: number
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.apply(createInputTensors([state]), {training: false}) as tf.Tensor | tf.Tensor[];
        const logitsHeads = parsePolicyOutput(predicted);
        
        // Sample actions from categorical distributions with optional noise
        const {actions, logitsFlat} = sampleActionsFromLogits(logitsHeads, noise);
        const logProb = computeLogProbCategorical(actions.expandDims(0), logitsHeads);

        noise?.forEach(t => t.dispose());

        return {
            logits: syncUnwrapTensor(logitsFlat) as Float32Array,
            actions: syncUnwrapTensor(actions) as Float32Array,
            logProb: syncUnwrapTensor(logProb)[0],
        };
    });
}

export function pureAct(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
): {
    actions: Float32Array,
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.apply(createInputTensors([state]), {training: false}) as tf.Tensor | tf.Tensor[];
        const logitsHeads = parsePolicyOutput(predicted);
        
        // Use argmax for deterministic action selection
        const actions = tf.concat(
            logitsHeads.map(logits => tf.argMax(logits, -1).expandDims(-1)),
            -1
        ).squeeze([0]);

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

export function computeVTraceTargets(
    policyNetwork: tf.LayersModel,
    valueNetwork: tf.LayersModel,
    batch: AgentMemoryBatch,
    batchSize: number,
    gamma: number,
    clipRho: number = 1,
    clipC: number = 1,
    clipRhoPG: number = 1,
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
    pureLogits: Float32Array,
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const predicted = policyNetwork.predict(input, {batchSize});
        const logitsHeads = parsePolicyOutput(predicted);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, ACTION_DIM]);
        const logProbCurrentTensor = computeLogProbCategorical(actions, logitsHeads);
        const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
        const rhosTensor = computeRho(logProbBehaviorTensor, logProbCurrentTensor);
        const valuesTensor = (valueNetwork.predict(input, {batchSize}) as tf.Tensor).squeeze();

        const values = syncUnwrapTensor(valuesTensor) as Float32Array;
        const rhos = syncUnwrapTensor(rhosTensor) as Float32Array;

        const {advantages, tdErrors, vTraces} = computeVTrace(
            batch.rewards,
            batch.dones,
            values,
            rhos,
            gamma,
            clipRho,
            clipC,
            clipRhoPG,
        );

        if (!arrayHealthCheck(advantages)) {
            throw new Error('VTrace advantages are NaN');
        }
        if (!arrayHealthCheck(tdErrors)) {
            throw new Error('VTrace tdErrors are NaN');
        }
        if (!arrayHealthCheck(vTraces)) {
            throw new Error('VTrace returns are NaN');
        }

        const normalizedAdvantages = normalize(advantages);
        
        // Flatten logits for logging
        const flatLogits = tf.concat(logitsHeads, -1);

        return {
            advantages: normalizedAdvantages,
            tdErrors: tdErrors,
            returns: vTraces,
            values: values,
            // just for logs
            pureLogits: syncUnwrapTensor(flatLogits) as Float32Array,
        };
    });
}

function computeRho(
    logProbBehavior: tf.Tensor,
    logProbCurrent: tf.Tensor,
): tf.Tensor {
    const logRho = logProbCurrent.sub(logProbBehavior);
    return logRho.exp(); // ρ_t = π_current / π_behavior
}

export function computeVTrace(
    rewards: Float32Array,
    dones: Float32Array,
    values: Float32Array,
    rhos: Float32Array,
    gamma: number,
    _clipRho: number,
    clipC: number,
    clipRhoPG: number,
): { vTraces: Float32Array, tdErrors: Float32Array, advantages: Float32Array } {
    const T = rewards.length;
    // bootstrap v̂_{T} = values[T]
    let nextVTrace = dones[T - 1] ? 0 : values[T];
    if (nextVTrace === undefined) {
        throw new Error('Implementation required last state as terminal');
    }

    const vTraces = new Float32Array(T);
    const tdErrors = new Float32Array(T);
    const advantages = new Float32Array(T);

    // bootstrap: v̂_T = done? 0 : V(s_T)
    let nextAdv = 0; // A_{t+1}

    for (let t = T - 1; t >= 0; --t) {
        const nextValue = dones[t] ? 0 : values[t + 1];
        const discount = dones[t] ? 0 : gamma;
        const value = values[t];

        const c = Math.min(rhos[t], clipC); // 0.95 * Math.min(rhos[t], clipC);
        const rho = 1; // Math.min(rhos[t], clipRho);
        const rhoPG = Math.min(rhos[t], clipRhoPG);

        // δ_t^V = r_t + γ V(s_{t+1}) - V(s_t)
        const tdError = rewards[t] + discount * nextValue - value;
        tdErrors[t] = tdError;

        // v̂_t = V(s_t) + ρ̄_t δ_t^V + γ c̄_t (v̂_{t+1} - V(s_{t+1}))
        vTraces[t] = value + rho * tdError + discount * c * (nextVTrace - nextValue);

        // A_t = ρ̄_t^{PG} δ_t^V + γ c̄_t A_{t+1}
        const adv = rhoPG * tdError + discount * c * nextAdv;
        advantages[t] = adv;

        nextVTrace = vTraces[t];
        nextAdv = adv;
    }

    return {vTraces, advantages, tdErrors};
}

function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[]): tf.Tensor[] {
    // Return array of logits tensors: [shootLogits, moveLogits, rotLogits, turRotLogits]
    return prediction as tf.Tensor[];
}

// Compute log probability for categorical distributions
function computeLogProbCategorical(actions: tf.Tensor, logitsHeads: tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
        const logProbs = logitsHeads.map((logits, i) => {
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
function computeEntropyCategorical(logitsHeads: tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
        const entropies = logitsHeads.map(logits => {
            const probs = tf.softmax(logits); // [B, numClasses]
            const logProbs = tf.logSoftmax(logits); // [B, numClasses]
            return probs.mul(logProbs).sum(-1).mul(-1); // -sum(p * log(p)) -> [B]
        });
        return tf.addN(entropies).div(logitsHeads.length).mean() as tf.Scalar; // Mean entropy across batch and heads
    });
}

// Sample actions from logits with optional noise
function sampleActionsFromLogits(
    logitsHeads: tf.Tensor[],
    noise?: tf.Tensor[]
): { actions: tf.Tensor, logitsFlat: tf.Tensor } {
    return tf.tidy(() => {
        const sampledActions = logitsHeads.map((logits, i) => {
            if (noise && noise[i]) {
                // Add provided noise (e.g., colored noise) to logits
                const noisyLogits = logits.add(noise[i]);
                return tf.argMax(noisyLogits, -1);
            } else {
                // Deterministic or random: use Gumbel noise if no specific noise provided
                if (noise === undefined) {
                    // Deterministic: argmax
                    return tf.argMax(logits, -1);
                } else {
                    // Random with Gumbel noise
                    const gumbel = tf.neg(tf.log(tf.neg(tf.log(tf.randomUniform(logits.shape, 1e-8, 1)))));
                    const noisyLogits = logits.add(gumbel);
                    return tf.argMax(noisyLogits, -1);
                }
            }
        });
        
        const actions = tf.stack(sampledActions, -1).squeeze([0]); // [actionDim]
        const logitsFlat = tf.concat(logitsHeads, -1).squeeze([0]); // Flatten for storage
        
        return { actions, logitsFlat };
    });
}

let randomInputTensors: tf.Tensor[];

function getRandomInputTensors() {
    randomInputTensors = randomInputTensors == null || random() > 0.9
        ? (tf.dispose(randomInputTensors), createInputTensors([prepareRandomInputArrays()]))
        : randomInputTensors;

    return randomInputTensors;
}

export function networkHealthCheck(network: tf.LayersModel): Promise<boolean> {
    const tData = tf.tidy(() => {
        const result = network.predict(getRandomInputTensors()) as tf.Tensor | tf.Tensor[];
        const arr = Array.isArray(result) ? result : [result];
        return arr.map(t => t.squeeze());
    });

    return onReadyRead()
        .then(() => Promise.all(tData.map(t => asyncUnwrapTensor(t))))
        .then(() => true);
}

