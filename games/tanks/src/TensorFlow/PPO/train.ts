import * as tf from '@tensorflow/tfjs';
import { dispose } from '@tensorflow/tfjs';
import { ACTION_DIM } from '../Common/consts.ts';
import { computeLogProb } from '../Common/computeLogProb.ts';
import { InputArrays, prepareRandomInputArrays } from '../Common/InputArrays.ts';
import { createInputTensors } from '../Common/InputTensors.ts';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { random } from '../../../../../lib/random.ts';

export function trainPolicyNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize]
    advantages: tf.Tensor,   // [batchSize]
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
): Promise<number> {
    const loss = tf.tidy(() => {
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
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            const entropyEachDim = logStd.add(c); // [batchSize,ACTION_DIM]
            const totalEntropy = entropyEachDim.sum(1).mean();
            const totalLoss = policyLoss.sub(totalEntropy.mul(entropyCoeff));

            return totalLoss as tf.Scalar;
        }, { clipNorm });
    });

    return loss.data()
        .then((v) => v[0])
        .finally(() => loss.dispose());
}

// Обучение сети критика (оценка состояний)
export async function trainValueNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
    oldValues: tf.Tensor, // [batchSize], для клиппинга
    clipRatio: number,
    clipNorm: number,
): Promise<number> {
    const loss = tf.tidy(() => {
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


    return loss.data()
        .then((v) => v[0])
        .finally(() => loss.dispose());
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
): Promise<number> {
    const value = tf.tidy(() => {
        const predict = policyNetwork.predict(states) as tf.Tensor;
        const { mean, logStd } = parsePredict(predict);
        const std = logStd.exp();
        const newLogProbs = computeLogProb(actions, mean, std);
        const diff = oldLogProb.sub(newLogProbs);
        const kl = diff.mean();
        return kl;
    });

    return value.data()
        .then((v) => v[0])
        .finally(() => value.dispose());
}

export function act(
    policyNetwork: tf.LayersModel,
    valueNetwork: tf.LayersModel,
    state: InputArrays,
): {
    actions: Float32Array,
    logProb: number,
    value: number
} {
    const result = tf.tidy(() => {
        const input = createInputTensors([state]);
        const predict = policyNetwork.predict(input) as tf.Tensor;
        const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
        const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук
        const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
        const clippedLogStd = outLogStd.clipByValue(-5, 0.2);
        const std = clippedLogStd.exp();
        const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        const actions = outMean.add(noise);
        const logProb = computeLogProb(actions, outMean, std);
        const value = valueNetwork.predict(input) as tf.Tensor;

        return {
            actions: actions.dataSync() as Float32Array,
            logProb: logProb.dataSync()[0],
            value: value.squeeze().dataSync()[0],
        };
    });

    if (!arrayHealthCheck(result.actions)) {
        throw new Error('Invalid actions data');
    }
    if (Number.isNaN(result.logProb)) {
        throw new Error('Invalid logProb data');
    }
    if (Number.isNaN(result.value)) {
        throw new Error('Invalid value data');
    }

    return result;
}

export function predict(policyNetwork: tf.LayersModel, state: InputArrays): { actions: Float32Array } {
    const result = tf.tidy(() => {
        const predict = policyNetwork.predict(createInputTensors([state])) as tf.Tensor;
        const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
        const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук

        // const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
        // const clippedLogStd = outLogStd.clipByValue(-5, 0.2);
        // const std = clippedLogStd.exp();
        // const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        // const actions = outMean.add(noise);

        return {
            // actions: actions.dataSync() as Float32Array,
            actions: outMean.dataSync() as Float32Array,
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

let randomInputTensors: tf.Tensor[];

function getRandomInputTensors() {
    randomInputTensors = randomInputTensors == null || random() > 0.9
        ? (dispose(randomInputTensors), createInputTensors([prepareRandomInputArrays()]))
        : randomInputTensors;

    return randomInputTensors;
}

export async function networkHealthCheck(network: tf.LayersModel): Promise<boolean> {
    const data = tf.tidy(() => {
        return (network.predict(getRandomInputTensors()) as tf.Tensor).squeeze().dataSync();
    }) as Float32Array;

    return arrayHealthCheck(data);
}

export function arrayHealthCheck(array: Float32Array): boolean {
    return array.every(Number.isFinite);
}