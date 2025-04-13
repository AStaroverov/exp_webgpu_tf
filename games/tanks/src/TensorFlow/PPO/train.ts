import * as tf from '@tensorflow/tfjs';
import { ACTION_DIM } from '../Common/consts.ts';
import { computeLogProb } from '../Common/computeLogProb.ts';
import { InputArrays, prepareRandomInputArrays } from '../Common/InputArrays.ts';
import { createInputTensors } from '../Common/InputTensors.ts';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { random } from '../../../../../lib/random.ts';
import { flatTypedArray } from '../Common/flat.ts';
import { normalize } from '../../../../../lib/math.ts';
import { Batch } from '../Common/Memory.ts';
import { CONFIG } from './config.ts';

export function trainPolicyNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize]
    advantages: tf.Tensor,   // [batchSize]
    weights: tf.Tensor,      // [batchSize], IS weights
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
): number {
    const tLoss = tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predict = network.predict(states) as tf.Tensor;
            const { mean, logStd } = parsePredict(predict);
            const std = logStd.exp();
            const newLogProbs = computeLogProb(actions, mean, std);
            const oldLogProbs2D = oldLogProbs.reshape(newLogProbs.shape);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs2D));
            const clippedRatio = ratio.clipByValue(1 - clipRatio, 1 + clipRatio);
            const surr1 = ratio.mul(advantages);
            const surr2 = clippedRatio.mul(advantages);
            const weightedMin = tf.minimum(surr1, surr2).mul(weights);
            const policyLoss = weightedMin.sum().div(weights.sum()).mul(-1);

            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            const entropyEachDim = logStd.add(c); // [batchSize,ACTION_DIM]
            const totalEntropy = entropyEachDim.sum(1).mean();
            const totalLoss = policyLoss.sub(totalEntropy.mul(entropyCoeff));

            return totalLoss as tf.Scalar;
        }, { clipNorm });
    });

    return unwrapTensor(tLoss)[0];
}

export function trainValueNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
    oldValues: tf.Tensor, // [batchSize], для клиппинга
    clipRatio: number,
    clipNorm: number,
): number {
    const tLoss = tf.tidy(() => {
        return optimize(network.optimizer, () => {
            // forward pass
            const predicted = network.predict(states) as tf.Tensor;
            // shape [batchSize,1], приводим к [batchSize]
            const valuePred = predicted.squeeze(); // [batchSize]

            // Клипаем (PPO2 style)
            const oldVal2D = oldValues.reshape(valuePred.shape);   // тоже [batchSize]
            const valuePredClipped = oldVal2D.add(
                valuePred.sub(oldVal2D).clipByValue(-clipRatio, clipRatio),
            );
            const returns2D = returns.reshape(valuePred.shape);

            const vfLoss1 = returns2D.sub(valuePred).square();
            const vfLoss2 = returns2D.sub(valuePredClipped).square();
            const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(0.5);

            return finalValueLoss as tf.Scalar;
        }, { clipNorm });
    });

    return unwrapTensor(tLoss)[0];
}


function parsePredict(predict: tf.Tensor) {
    const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
    const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
    const clippedLogStd = outLogStd.clipByValue(-5, 0.2);

    return { mean: outMean, logStd: clippedLogStd };
}

export function computeKullbackLeibler(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,
    oldLogProb: tf.Tensor,
): number {
    return tf.tidy(() => {
        const predict = policyNetwork.predict(states) as tf.Tensor;
        const { mean, logStd } = parsePredict(predict);
        const std = logStd.exp();
        const newLogProbs = computeLogProb(actions, mean, std);
        const diff = oldLogProb.sub(newLogProbs);
        const kl = diff.mean();
        return kl.dataSync()[0];
    });
}

export function act(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
): {
    actions: Float32Array,
    logProb: number,
} {
    const result = tf.tidy(() => {
        const { mean, std } = getMeanAndStd(policyNetwork, 1, createInputTensors([state]));

        const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        const actions = mean.add(noise);
        const logProb = computeLogProb(actions, mean, std);

        return {
            actions: actions.dataSync() as Float32Array,
            logProb: logProb.dataSync()[0],
        };
    });

    if (!arrayHealthCheck(result.actions)) {
        throw new Error('Invalid actions data');
    }
    if (Number.isNaN(result.logProb)) {
        throw new Error('Invalid logProb data');
    }

    return result;
}

export function predict(policyNetwork: tf.LayersModel, state: InputArrays): { actions: Float32Array } {
    const result = tf.tidy(() => {
        const { mean } = getMeanAndStd(policyNetwork, 1, createInputTensors([state]));

        // const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        // const actions = outMean.add(noise);

        return {
            // actions: actions.dataSync() as Float32Array,
            actions: mean.dataSync() as Float32Array,
        };
    });

    if (!arrayHealthCheck(result.actions)) {
        throw new Error('Invalid actions data');
    }

    return result;
}

function optimize(
    optimizer: tf.Optimizer,
    predict: () => Scalar,
    options?: { clipNorm?: number },
): tf.Scalar {
    const clipNorm = options?.clipNorm ?? 1;

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
        const clippedGrads: Record<string, tf.Tensor> = {};
        for (const [varName, grad] of Object.entries(grads)) {
            clippedGrads[varName] = tf.mul(grad, clipCoef);
        }

        // применяем обрезанные градиенты
        optimizer.applyGradients(clippedGrads);

        tf.dispose(grads);

        return value;
    });
}

function getMeanAndStd(
    policyNetwork: tf.LayersModel,
    batchSize: number,
    input: tf.Tensor[],
): {
    mean: tf.Tensor,
    std: tf.Tensor,
} {
    const raw = policyNetwork.predict(input) as tf.Tensor;
    const mean = raw.slice([0, 0], [batchSize, ACTION_DIM]);   // ACTION_DIM штук
    const outLogStd = raw.slice([0, ACTION_DIM], [batchSize, ACTION_DIM]);
    const clippedLogStd = outLogStd.clipByValue(-5, 0.2);
    const std = clippedLogStd.exp();
    return { mean, std };
}

export function computeVTraceTargets(
    policyNetwork: tf.LayersModel,
    valueNetwork: tf.LayersModel,
    batch: Batch,
    gamma: number = CONFIG.gamma,
    clipRho: number = 1.0,
    clipC: number = 1.0,
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const { mean: meanCurrent, std: stdCurrent } = getMeanAndStd(policyNetwork, batch.size, input);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, ACTION_DIM]);

        const logProbCurrentTensor = computeLogProb(actions, meanCurrent, stdCurrent);
        const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
        const rhosTensor = computeRho(logProbBehaviorTensor, logProbCurrentTensor);
        const valuesTensor = (valueNetwork.predict(input) as tf.Tensor).squeeze();

        const values = (valuesTensor.dataSync()) as Float32Array;
        const rhos = (rhosTensor.dataSync()) as Float32Array;

        const { vTraces: returns, tdErrors } = computeVTrace(
            batch.rewards,
            batch.dones,
            values,
            rhos,
            gamma,
            clipRho,
            clipC,
        );
        // 8) Считаем advantages = vs[t] - values[t]
        const advantages = returns.map((v, i) => v - values[i]);

        return {
            advantages: normalize(advantages),
            tdErrors: tdErrors,
            returns: returns,
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

function computeVTrace(
    rewards: Float32Array,
    dones: Float32Array,
    values: Float32Array,
    rhos: Float32Array,
    gamma: number,
    clipRho: number,
    clipC: number,
): { vTraces: Float32Array, tdErrors: Float32Array } {
    const batchSize = values.length;
    // Соберём nextValuesArr:
    const nextValues = createNextValues(values, dones);
    // 7) Делаем цикл V-trace
    const vTraces = new Float32Array(batchSize);
    const tdErrors = new Float32Array(batchSize);

    // Инициализируем последний элемент
    vTraces[batchSize - 1] = values[batchSize - 1];

    for (let t = batchSize - 2; t >= 0; t--) {
        // Если done[t], сбрасываем V-trace именно на valuesArr[t]
        // (т.е. "начинаем" новый эпизод с самого себя)
        if (dones[t] === 1) {
            vTraces[t] = values[t];
            continue;
        }

        // c_t = min(rho_t, clipC)
        const c_t = Math.min(rhos[t], clipC);
        // rho_t = min(rho_t, clipRho)
        const rho_t = Math.min(rhos[t], clipRho);

        // Если done[t], то discount=0, иначе gamma
        const discount = dones[t] === 1 ? 0 : gamma;

        const tdError = (
            rewards[t]
            + discount * nextValues[t]
            - values[t]
        );
        // delta_t
        const delta_t = rho_t * tdError;

        // vs[t]
        vTraces[t] = values[t]
            + delta_t
            + discount * c_t * (vTraces[t + 1] - nextValues[t]);
        tdErrors[t] = tdError;
    }

    return { vTraces, tdErrors };
}

function createNextValues(values: Float32Array, dones: Float32Array): Float32Array {
    // Для nextValues, обычно делаем сдвиг на 1, но нужно аккуратно учесть done.
    // Самый простой путь:
    //   values[i+1] -> nextValue для шага i
    //   Если done[i], то nextValue = 0

    const batchSize = values.length;
    const nextValues = new Float32Array(batchSize);
    for (let i = 0; i < batchSize - 1; i++) {
        // Если done на шаге i, то следующий state – терминальный => nextValue=0
        nextValues[i] = dones[i] === 1 ? 0 : values[i + 1];
    }
    // Для последнего шага
    nextValues[batchSize - 1] = dones[batchSize - 1] === 1 ? 0 : values[batchSize - 1];

    return nextValues;
}

let randomInputTensors: tf.Tensor[];

function getRandomInputTensors() {
    randomInputTensors = randomInputTensors == null || random() > 0.9
        ? (tf.dispose(randomInputTensors), createInputTensors([prepareRandomInputArrays()]))
        : randomInputTensors;

    return randomInputTensors;
}

export function networkHealthCheck(network: tf.LayersModel): boolean {
    const data = tf.tidy(() => {
        return (network.predict(getRandomInputTensors()) as tf.Tensor).squeeze().dataSync();
    }) as Float32Array;

    return arrayHealthCheck(data);
}

export function arrayHealthCheck(array: Float32Array | Uint8Array | Int32Array): boolean {
    return array.every(Number.isFinite);
}

export function unwrapTensor<T extends Float32Array | Uint8Array | Int32Array>(tensor: tf.Tensor): T {
    try {
        const value = tensor.dataSync() as T;
        if (!arrayHealthCheck(value)) {
            throw new Error('Invalid loss value');
        }
        return value;
    } finally {
        tensor.dispose();
    }
}