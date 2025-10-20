// SAC Training functions
import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import { sampleActionWithTanhSquashing } from '../../../ml-common/computeLogProb.ts';
import { InputArrays } from '../../../ml-common/InputArrays.ts';
import { createInputTensors } from '../../../ml-common/InputTensors.ts';
import { syncUnwrapTensor } from '../../../ml-common/Tensor.ts';

/**
 * Parse policy network output and apply log_std bounds
 */
function parsePredict(predict: tf.Tensor[], minLogStd: number, maxLogStd: number) {
    const outMean: tf.Tensor = predict[0];
    const outLogStd: tf.Tensor = predict[1];

    let clippedLogStd = outLogStd;

    if (isFinite(minLogStd) && isFinite(maxLogStd)) {
        // Hard clipping for stability
        clippedLogStd = outLogStd.clipByValue(minLogStd, maxLogStd);
    }

    return {
        mean: outMean,
        logStd: clippedLogStd,
        pureLogStd: outLogStd,
    };
}

/**
 * Optimizer wrapper with gradient clipping
 */
function optimize(
    optimizer: tf.Optimizer,
    predict: () => Scalar,
    options?: { returnCost?: boolean, clipNorm?: number },
): undefined | tf.Scalar {
    const clipNorm = options?.clipNorm ?? 1;
    const returnCost = options?.returnCost ?? false;

    return tf.tidy(() => {
        const { grads, value } = tf.variableGrads(predict);

        // Compute global gradient norm
        const gradsArray = Object.values(grads).map(g => g.square().sum());
        const sumSquares = gradsArray.reduce((acc, t) => acc.add(t), tf.scalar(0));
        const globalNorm = sumSquares.sqrt();

        // Compute clipping coefficient
        const eps = 1e-8;
        const safeGlobalNorm = tf.maximum(globalNorm, tf.scalar(eps));
        const clipCoef = tf.minimum(tf.scalar(1), tf.div(clipNorm, safeGlobalNorm));

        // Apply clipping to each gradient
        const clippedGrads: NamedTensor[] = [];
        for (const [varName, grad] of Object.entries(grads)) {
            clippedGrads.push({ name: varName, tensor: tf.mul(grad, clipCoef) });
        }

        // Fix for internal implementation of applyGradients
        clippedGrads.sort((a, b) => a.name.localeCompare(b.name));

        // Apply clipped gradients
        optimizer.applyGradients(clippedGrads);

        tf.dispose(grads);

        return returnCost ? value : undefined;
    });
}

/**
 * Train Actor Network (Policy)
 * Maximizes: E[Q(s,a) - α * log π(a|s)]
 */
export function trainActorNetwork(
    actorNetwork: tf.LayersModel,
    criticNetwork1: tf.LayersModel,
    criticNetwork2: tf.LayersModel,
    states: tf.Tensor[],
    alpha: number,
    batchSize: number,
    clipNorm: number,
    minLogStd: number,
    maxLogStd: number,
    returnCost: boolean,
): undefined | tf.Tensor {
    return tf.tidy(() => {
        return optimize(actorNetwork.optimizer, () => {
            // 1. Predict mean and log_std from actor
            const predicted = actorNetwork.predict(states, { batchSize }) as tf.Tensor[];
            const { mean, logStd } = parsePredict(predicted, minLogStd, maxLogStd);

            // 2. Sample actions with reparameterization trick
            const batchShape = mean.shape[0];
            const actionDim = mean.shape[1] || 1;
            const epsilon = tf.randomNormal([batchShape, actionDim]);
            const { action, logProb } = sampleActionWithTanhSquashing(mean, logStd, epsilon);

            // 3. Compute Q-values from both critics
            const q1 = (criticNetwork1.predict([...states, action], { batchSize }) as tf.Tensor).squeeze();
            const q2 = (criticNetwork2.predict([...states, action], { batchSize }) as tf.Tensor).squeeze();
            const minQ = tf.minimum(q1, q2);

            // 4. Actor loss: E[α * log π(a|s) - Q(s,a)]
            // We want to maximize Q - α*logProb, so minimize -(Q - α*logProb)
            const actorLoss = tf.scalar(alpha).mul(logProb).sub(minQ).mean();

            return actorLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });
}

/**
 * Train Critic Networks (Q-functions)
 * Minimizes: E[(Q(s,a) - target)²] where target = r + γ * (min Q_target(s',a') - α * log π(a'|s'))
 */
export function trainCriticNetworks(
    critic1: tf.LayersModel,
    critic2: tf.LayersModel,
    targetCritic1: tf.LayersModel,
    targetCritic2: tf.LayersModel,
    actorNetwork: tf.LayersModel,
    batch: {
        states: tf.Tensor[],
        actions: tf.Tensor,
        rewards: tf.Tensor,
        nextStates: tf.Tensor[],
        dones: tf.Tensor
    },
    alpha: number,
    gamma: number,
    batchSize: number,
    clipNorm: number,
    minLogStd: number,
    maxLogStd: number,
    returnCost: boolean,
): { loss1: tf.Tensor | undefined, loss2: tf.Tensor | undefined } {
    const { states, actions, rewards, nextStates, dones } = batch;

    // 1. Compute target Q-value (no gradient through this)
    const targetQ = tf.tidy(() => {
        // Sample next actions from current policy
        const predictedNext = actorNetwork.predict(nextStates, { batchSize }) as tf.Tensor[];
        const { mean: nextMean, logStd: nextLogStd } = parsePredict(predictedNext, minLogStd, maxLogStd);

        const batchShape = nextMean.shape[0];
        const actionDim = nextMean.shape[1] || 1;
        const epsilon = tf.randomNormal([batchShape, actionDim]);
        const { action: nextAction, logProb: nextLogProb } = sampleActionWithTanhSquashing(
            nextMean, nextLogStd, epsilon
        );

        // Compute target Q-values
        const targetQ1 = (targetCritic1.predict([...nextStates, nextAction], { batchSize }) as tf.Tensor).squeeze();
        const targetQ2 = (targetCritic2.predict([...nextStates, nextAction], { batchSize }) as tf.Tensor).squeeze();
        const minTargetQ = tf.minimum(targetQ1, targetQ2);

        // Bellman backup with entropy regularization
        // target = r + γ * (1 - done) * (Q_target(s', a') - α * log π(a'|s'))
        const target = rewards.add(
            tf.scalar(gamma).mul(
                tf.scalar(1).sub(dones).mul(
                    minTargetQ.sub(tf.scalar(alpha).mul(nextLogProb))
                )
            )
        );

        // Stop gradient to prevent backprop through target
        return tf.keep(target);
    });

    const targetQDetached = targetQ;

    // 2. Train Critic 1
    const loss1 = tf.tidy(() => {
        return optimize(critic1.optimizer, () => {
            const currentQ1 = (critic1.predict([...states, actions], { batchSize }) as tf.Tensor).squeeze();
            const criticLoss = tf.losses.meanSquaredError(targetQDetached, currentQ1).mean();
            return criticLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });

    // 3. Train Critic 2
    const loss2 = tf.tidy(() => {
        return optimize(critic2.optimizer, () => {
            const currentQ2 = (critic2.predict([...states, actions], { batchSize }) as tf.Tensor).squeeze();
            const criticLoss = tf.losses.meanSquaredError(targetQDetached, currentQ2).mean();
            return criticLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });

    // Cleanup
    targetQ.dispose();

    return { loss1, loss2 };
}

/**
 * Train temperature parameter α (optional auto-tuning)
 * Minimizes: E[-α * (log π(a|s) + target_entropy)]
 */
export function trainTemperature(
    alphaVariable: tf.Variable,
    logProbs: tf.Tensor,
    targetEntropy: number,
    learningRate: number,
    clipNorm: number,
    returnCost: boolean,
): tf.Tensor | undefined {
    const optimizer = tf.train.adam(learningRate);

    return tf.tidy(() => {
        return optimize(optimizer, () => {
            // Alpha loss: -α * (log π + target_entropy)
            // Minimize to increase α when entropy is below target
            const alphaLoss = alphaVariable.mul(
                logProbs.add(tf.scalar(targetEntropy)).mean()
            ).mul(-1);

            return alphaLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });
}

/**
 * Sample action from policy for inference
 */
export function act(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
    minLogStd: number,
    maxLogStd: number,
    deterministic: boolean = false,
): {
    actions: Float32Array,
    mean: Float32Array,
    logStd: Float32Array,
    logProb: number
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(createInputTensors([state])) as tf.Tensor[];
        const { mean, logStd } = parsePredict(predicted, minLogStd, maxLogStd);

        let actions: tf.Tensor;
        let logProb: tf.Tensor;

        if (deterministic) {
            // Use mean action for evaluation
            actions = tf.tanh(mean);
            logProb = tf.scalar(0); // Not used in deterministic mode
        } else {
            // Sample with reparameterization
            const actionDim = mean.shape[1] || 1;
            const epsilon = tf.randomNormal([1, actionDim]);
            const result = sampleActionWithTanhSquashing(mean, logStd, epsilon);
            actions = result.action;
            logProb = result.logProb;
        }

        return {
            actions: syncUnwrapTensor(actions) as Float32Array,
            mean: syncUnwrapTensor(mean) as Float32Array,
            logStd: syncUnwrapTensor(logStd) as Float32Array,
            logProb: syncUnwrapTensor(logProb)[0],
        };
    });
}
