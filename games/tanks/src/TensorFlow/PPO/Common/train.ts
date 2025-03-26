import * as tf from '@tensorflow/tfjs';
import { ACTION_DIM } from '../../Common/consts.ts';
import { computeLogProb } from '../../Common/computeLogProb.ts';
import { Config } from './config.ts';

export function trainPolicyNetwork(
    policyNetwork: tf.LayersModel,
    policyOptimizer: tf.Optimizer,
    config: Config,
    states: tf.Tensor,       // [batchSize, inputDim]
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize]
    advantages: tf.Tensor,   // [batchSize]
): number {
    return tf.tidy(() => {
        const totalLoss = policyOptimizer.minimize(() => {
            const predict = policyNetwork.predict(states) as tf.Tensor;
            const { mean, logStd } = parsePredict(predict);
            const std = logStd.exp();
            const newLogProbs = computeLogProb(actions, mean, std);
            const oldLogProbs2D = oldLogProbs.reshape(newLogProbs.shape);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs2D));
            const clippedRatio = ratio.clipByValue(1 - config.clipRatioPolicy, 1 + config.clipRatioPolicy);
            const surr1 = ratio.mul(advantages);
            const surr2 = clippedRatio.mul(advantages);
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            const entropyEachDim = logStd.add(c); // [batchSize,ACTION_DIM]
            const totalEntropy = entropyEachDim.sum(1).mean();
            const totalLoss = policyLoss.sub(totalEntropy.mul(config.entropyCoeff));

            return totalLoss as tf.Scalar;
        }, true);

        if (totalLoss == null) {
            throw new Error('Policy loss is null');
        }

        // Возвращаем число
        return totalLoss!.dataSync()[0];
    });
}

function parsePredict(predict: tf.Tensor) {
    const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
    const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
    const clippedLogStd = outLogStd.clipByValue(-2, 0.2);

    return { mean: outMean, logStd: clippedLogStd };
}

export function computeKullbackLeibler(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor,
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
        return kl.arraySync() as number;
    });
}

// Обучение сети критика (оценка состояний)
export function trainValueNetwork(
    valueNetwork: tf.LayersModel,
    valueOptimizer: tf.Optimizer,
    config: Config,
    states: tf.Tensor,   // [batchSize, inputDim]
    returns: tf.Tensor,  // [batchSize], уже подсчитанные (GAE + V(s) или просто discountedReturns)
    oldValues: tf.Tensor, // [batchSize], для клиппинга
): number {
    return tf.tidy(() => {
        const vfLoss = valueOptimizer.minimize(() => {
            // forward pass
            const predicted = valueNetwork.predict(states) as tf.Tensor;
            // shape [batchSize,1], приводим к [batchSize]
            const valuePred = predicted.squeeze(); // [batchSize]

            // Клипаем (PPO2 style)
            const oldVal2D = oldValues.reshape(valuePred.shape);   // тоже [batchSize]
            const valuePredClipped = oldVal2D.add(
                valuePred.sub(oldVal2D).clipByValue(-config.clipRatioValue, config.clipRatioValue),
            );
            const returns2D = returns.reshape(valuePred.shape);

            const vfLoss1 = returns2D.sub(valuePred).square();
            const vfLoss2 = returns2D.sub(valuePredClipped).square();
            const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(0.5);

            return finalValueLoss as tf.Scalar;
        }, true);

        if (vfLoss == null) {
            throw new Error('Value loss is null');
        }

        return vfLoss!.dataSync()[0];
    });
}

export function act(
    policyNetwork: tf.LayersModel,
    valueNetwork: tf.LayersModel,
    state: Float32Array,
): {
    actions: Float32Array,
    logProb: number,
    value: number
} {
    return tf.tidy(() => {
        const stateTensor = tf.tensor1d(state).expandDims(0);
        const predict = policyNetwork.predict(stateTensor) as tf.Tensor;
        const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
        const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук
        const outLogStd = rawOutputSqueezed.slice([ACTION_DIM], [ACTION_DIM]);
        const clippedLogStd = outLogStd.clipByValue(-2, 0.2);
        const std = clippedLogStd.exp();
        const noise = tf.randomNormal([ACTION_DIM]).mul(std);
        const actions = outMean.add(noise);
        const logProb = computeLogProb(actions, outMean, std);
        const value = valueNetwork.predict(stateTensor) as tf.Tensor;

        return {
            actions: actions.dataSync() as Float32Array,
            logProb: logProb.dataSync()[0],
            value: value.squeeze().dataSync()[0],
        };
    });
}

export function predict(policyNetwork: tf.LayersModel, state: Float32Array): { action: Float32Array } {
    return tf.tidy(() => {
        const stateTensor = tf.tensor1d(state).expandDims(0);
        const predict = policyNetwork.predict(stateTensor) as tf.Tensor;
        const rawOutputSqueezed = predict.squeeze(); // [ACTION_DIM * 2] при batch=1
        const outMean = rawOutputSqueezed.slice([0], [ACTION_DIM]);   // ACTION_DIM штук

        return {
            action: outMean.dataSync() as Float32Array,
        };
    });
}
