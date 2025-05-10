import * as tf from '@tensorflow/tfjs';
import { ACTION_DIM } from '../Common/consts.ts';
import { computeLogProb } from '../Common/computeLogProb.ts';
import { InputArrays, prepareRandomInputArrays } from '../Common/InputArrays.ts';
import { createInputTensors } from '../Common/InputTensors.ts';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { random } from '../../../../../lib/random.ts';
import { flatTypedArray } from '../Common/flat.ts';
import { normalize } from '../../../../../lib/math.ts';
import { AgentMemoryBatch } from '../Common/Memory.ts';
import { CONFIG } from './config.ts';
import { arrayHealthCheck, asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../Common/Tensor.ts';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';

const PREDICT_OPTIONS = { batchSize: CONFIG.miniBatchSize };

export function trainPolicyNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize]
    advantages: tf.Tensor,   // [batchSize]
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
    returnCost: boolean,
): undefined | tf.Tensor {
    return tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predicted = network.predict(states, PREDICT_OPTIONS) as tf.Tensor;
            const { mean, logStd } = parsePredict(predicted);
            const std = logStd.exp();
            const newLogProbs = computeLogProb(actions, mean, std);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
            const clippedRatio = ratio.clipByValue(1 - clipRatio, 1 + clipRatio);
            const surr1 = ratio.mul(advantages);
            const surr2 = clippedRatio.mul(advantages);
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            const entropyEachDim = logStd.add(c); // [batchSize,ACTION_DIM]
            const totalEntropy = entropyEachDim.sum(1).mean();
            const totalLoss = policyLoss.sub(totalEntropy.mul(entropyCoeff));

            return totalLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });
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
            const newValues = (network.predict(states, PREDICT_OPTIONS) as tf.Tensor).squeeze();
            const newValuesClipped = oldValues.add(
                newValues.sub(oldValues).clipByValue(-clipRatio, clipRatio),
            );

            const vfLoss1 = returns.sub(newValues).square();
            const vfLoss2 = returns.sub(newValuesClipped).square();
            const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(lossCoeff);

            return finalValueLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });
}

export function computeKullbackLeiblerAprox(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,
    oldLogProb: tf.Tensor,
): number {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, PREDICT_OPTIONS) as tf.Tensor;
        const { mean, logStd } = parsePredict(predicted);
        const std = logStd.exp();
        const newLogProbs = computeLogProb(actions, mean, std);
        const diff = oldLogProb.sub(newLogProbs);
        const kl = diff.mean();
        return kl.dataSync()[0];
    });
}

export function computeKullbackLeiblerExact(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    oldMean: tf.Tensor,
    oldLogStd: tf.Tensor,
): tf.Tensor {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, PREDICT_OPTIONS) as tf.Tensor;
        const { mean: newMean, logStd: newLogStd } = parsePredict(predicted);

        const oldStd = oldLogStd.exp();
        const newStd = newLogStd.exp();

        const numerator = oldStd.square().add(oldMean.sub(newMean).square());
        const denominator = newStd.square().mul(2);
        const logTerm = newLogStd.sub(oldLogStd); // log(σ₂ / σ₁)

        const klPerDim = logTerm.add(numerator.div(denominator)).sub(0.5);
        const kl = klPerDim.sum(1).mean();

        return kl;
    });
}

export function act(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
): {
    actions: Float32Array,
    mean: Float32Array,
    logStd: Float32Array,
    logProb: number
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(createInputTensors([state]), PREDICT_OPTIONS) as tf.Tensor;
        const { mean, logStd } = parsePredict(predicted);
        const std = logStd.exp();

        const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        const actions = mean.add(noise);
        const logProb = computeLogProb(actions, mean, std);

        return {
            actions: syncUnwrapTensor(actions) as Float32Array,
            mean: syncUnwrapTensor(mean) as Float32Array,
            logStd: syncUnwrapTensor(logStd) as Float32Array,
            logProb: syncUnwrapTensor(logProb)[0],
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
        const { grads, value } = tf.variableGrads(predict);

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

export function computeVTraceTargets(
    policyNetwork: tf.LayersModel,
    valueNetwork: tf.LayersModel,
    batch: AgentMemoryBatch,
    gamma: number = CONFIG.gamma,
    clipRho: number = 1.0,
    clipC: number = 1.0,
    clipRhoPG: number = 1.0,
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const predicted = policyNetwork.predict(input, PREDICT_OPTIONS) as tf.Tensor;
        const { mean: meanCurrent, logStd: logStdCurrent } = parsePredict(predicted);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, ACTION_DIM]);

        const logProbCurrentTensor = computeLogProb(actions, meanCurrent, logStdCurrent.exp());
        const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
        const rhosTensor = computeRho(logProbBehaviorTensor, logProbCurrentTensor);
        const valuesTensor = (valueNetwork.predict(input, PREDICT_OPTIONS) as tf.Tensor).squeeze();

        const values = syncUnwrapTensor(valuesTensor) as Float32Array;
        const rhos = syncUnwrapTensor(rhosTensor) as Float32Array;

        const { advantages, tdErrors, vTraces } = computeVTrace(
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
        if (!arrayHealthCheck(values)) {
            throw new Error('VTrace values are NaN');
        }

        return {
            advantages: normalize(advantages),
            tdErrors: tdErrors,
            returns: vTraces,
            values: values,
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
    clipRho: number,
    clipC: number,
    clipRhoPG: number,
): { vTraces: Float32Array, tdErrors: Float32Array, advantages: Float32Array } {
    const T = rewards.length;
    const vTraces = new Float32Array(T);
    const tdErrors = new Float32Array(T);
    const advantages = new Float32Array(T);

    // bootstrap v̂_{T} = values[T]
    let vtp1 = dones[T - 1] ? 0 : values[T];

    if (vtp1 === undefined) {
        throw new Error('Implementation required last state as terminal');
    }

    for (let t = T - 1; t >= 0; --t) {
        const discount = dones[t] ? 0 : gamma;
        const nextValue = dones[t] ? 0 : values[t + 1];
        const value = values[t];

        const rho = Math.min(rhos[t], clipRho);
        const c = Math.min(rhos[t], clipC);

        const tdError = (rewards[t] + discount * nextValue - value);
        tdErrors[t] = tdError;
        const delta = rho * tdError;
        vTraces[t] = value + delta + discount * c * (vtp1 - nextValue);

        // policy-gradient advantage
        const rhoPG = Math.min(rhos[t], clipRhoPG);
        advantages[t] = rhoPG * (rewards[t] + discount * vtp1 - value);

        vtp1 = vTraces[t];
    }

    return { vTraces, advantages, tdErrors };
}

function parsePredict(predict: tf.Tensor) {
    const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
    const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
    const clippedLogStd = outLogStd.clipByValue(-5, 0.2);

    return { mean: outMean, logStd: clippedLogStd };
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
        return (network.predict(getRandomInputTensors(), PREDICT_OPTIONS) as tf.Tensor).squeeze();
    });

    return onReadyRead().then(() => asyncUnwrapTensor(tData)).then(() => true);
}
