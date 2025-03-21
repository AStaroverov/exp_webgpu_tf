import * as tf from '@tensorflow/tfjs';
import { ACTION_DIM } from '../../Common/consts.ts';
import { computeLogProbTanh } from '../../Common/computeLogProb.ts';
import { isDevtoolsOpen } from '../../Common/utils.ts';
import { abs } from '../../../../../../lib/math.ts';
import { Config } from './config.ts';

export function trainPolicyNetwork(
    policyNetwork: tf.LayersModel,
    policyOptimizer: tf.Optimizer,
    config: Config,
    states: tf.Tensor,       // [batchSize, inputDim]
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize] или [batchSize,1]
    advantages: tf.Tensor,   // [batchSize]
): number {
    return tf.tidy(() => {
        const totalLoss = policyOptimizer.minimize(() => {
            const predict = policyNetwork.predict(states) as tf.Tensor;
            const outMean = predict.slice([0, 0], [-1, ACTION_DIM]);
            const outLogStd = predict.slice([0, ACTION_DIM], [-1, ACTION_DIM]);
            const clippedLogStd = outLogStd.clipByValue(-2, 0.2);
            const std = clippedLogStd.exp();
            const newLogProbs = computeLogProbTanh(actions, outMean, std);
            const oldLogProbs2D = oldLogProbs.reshape(newLogProbs.shape);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs2D));
            isDevtoolsOpen() && console.log('>> RATIO SUM ABS DELTA', (ratio.dataSync() as Float32Array).reduce((a, b) => a + abs(1 - b), 0));

            const surr1 = ratio.mul(advantages);
            const clippedRatio = ratio.clipByValue(1 - config.clipRatioPolicy, 1 + config.clipRatioPolicy);
            const surr2 = clippedRatio.mul(advantages);
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);

            const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            const entropyEachDim = clippedLogStd.add(c); // [batchSize,ACTION_DIM]
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
