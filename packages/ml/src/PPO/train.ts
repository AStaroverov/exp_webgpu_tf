import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import { normalize } from '../../../../lib/math.ts';
import { random } from '../../../../lib/random.ts';
import { computeLogProbTanh } from '../../../ml-common/computeLogProb.ts';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import { flatTypedArray } from '../../../ml-common/flat.ts';
import { InputArrays, prepareRandomInputArrays } from '../../../ml-common/InputArrays.ts';
import { createInputTensors } from '../../../ml-common/InputTensors.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';
import { NoiseMatrix } from '../../../ml-common/NoiseMatrix.ts';
import { arrayHealthCheck, asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../../../ml-common/Tensor.ts';

// let tC: undefined | tf.Tensor

export function trainPolicyNetwork(
    network: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,      // [batchSize, actionDim]
    oldLogProbs: tf.Tensor,  // [batchSize]
    advantages: tf.Tensor,   // [batchSize]
    batchSize: number,
    clipRatio: number,
    entropyCoeff: number,
    clipNorm: number,
    minLogStd: number[],
    maxLogStd: number[],
    returnCost: boolean,
): undefined | tf.Tensor {
    return tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predicted = network.predict(states, { batchSize });
            const { mean, logStd, phi } = parsePolicyOutput(predicted, minLogStd, maxLogStd);
            const { stdEff } = computeEffectiveStd(logStd, phi);
            const newLogProbs = computeLogProbTanh(actions, mean, stdEff);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
            const clippedRatio = ratio.clipByValue(1 - clipRatio, 1 + clipRatio);
            const surr1 = ratio.mul(advantages);
            const surr2 = clippedRatio.mul(advantages);
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);
            // const c = 0.5 * Math.log(2 * Math.PI * Math.E);
            // entropy = clippedLogStdEff.add(c).sum(1).mean().mul(entropyCoeff);
            const totalLoss = policyLoss;//.sub(entropy);
            return totalLoss as tf.Scalar;
        }, { clipNorm, returnCost });
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
            const newValues = (network.predict(states, { batchSize }) as tf.Tensor).squeeze();
            const newValuesClipped = oldValues.add(
                newValues.sub(oldValues).clipByValue(-clipRatio, clipRatio),
            );

            // mse with clipping
            // const vfLoss1 = returns.sub(newValues).square();
            // const vfLoss2 = returns.sub(newValuesClipped).square();

            // huber loss with clipping
            const rMean = returns.mean();
            const rStd = returns.sub(rMean).square().mean().sqrt().add(1e-8);
            const e1 = returns.sub(newValues);
            const e2 = returns.sub(newValuesClipped);
            const vfLoss1 = huber(e1, rStd);
            const vfLoss2 = huber(e2, rStd);

            const finalValueLoss = tf.maximum(vfLoss1, vfLoss2).mean().mul(lossCoeff) as tf.Scalar;
            return finalValueLoss as tf.Scalar;
        }, { clipNorm, returnCost });
    });
}

function huber(e: tf.Tensor, delta: number | tf.Tensor): tf.Tensor {
    const d = (typeof delta === 'number') ? tf.scalar(delta, 'float32') : delta as tf.Tensor;
    const abs = e.abs();
    const m = tf.minimum(abs, d);                     // no boolean ops
    // 0.5*m^2 + d*(abs - m)
    return m.square().mul(0.5).add(abs.sub(m).mul(d));
}

export function computeKullbackLeiblerAprox(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    actions: tf.Tensor,
    oldLogProb: tf.Tensor,
    batchSize: number,
    minLogStd: number[],
    maxLogStd: number[],
) {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, { batchSize });
        const { mean, logStd, phi } = parsePolicyOutput(predicted, minLogStd, maxLogStd);
        const { stdEff } = computeEffectiveStd(logStd, phi);
        const newLogProbs = computeLogProbTanh(actions, mean, stdEff);
        const diff = oldLogProb.sub(newLogProbs);
        const kl = diff.mean().abs();
        return kl;
    });
}

export function noisyAct(
    policyNetwork: tf.LayersModel,
    state: InputArrays,
    minLogStd: number[],
    maxLogStd: number[],
    noiseMatrix?: NoiseMatrix,
): {
    actions: Float32Array,
    mean: Float32Array,
    logProb: number
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(createInputTensors([state]));
        const { mean, logStd, phi } = parsePolicyOutput(predicted, minLogStd, maxLogStd);
        const { stdEff } = computeEffectiveStd(logStd, phi);
        const eps = noiseMatrix?.noise(logStd, phi);
        const actions = (eps ? mean.add(eps) : mean).tanh();
        const logProb = computeLogProbTanh(actions, mean, stdEff);

        return {
            actions: syncUnwrapTensor(actions) as Float32Array,
            mean: syncUnwrapTensor(mean) as Float32Array,
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
        const predicted = policyNetwork.predict(createInputTensors([state]));
        const { mean } = parsePolicyOutput(predicted, [0], [0]);
        const actions = mean.tanh();

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
    batchSize: number,
    gamma: number,
    minLogStd: number[],
    maxLogStd: number[],
    clipRho: number = 1,
    clipC: number = 1,
    clipRhoPG: number = 1,
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
    pureMean: Float32Array,
    pureLogStd: Float32Array,
    logStdShift: number,
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const predicted = policyNetwork.predict(input, { batchSize });
        const { mean, logStd, phi } = parsePolicyOutput(predicted, minLogStd, maxLogStd);
        const { stdEff, pureLogStdEff, power: phiPower } = computeEffectiveStd(logStd, phi);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, ACTION_DIM]);
        const logProbCurrentTensor = computeLogProbTanh(actions, mean, stdEff);
        const logProbBehaviorTensor = tf.tensor1d(batch.logProbs);
        const rhosTensor = computeRho(logProbBehaviorTensor, logProbCurrentTensor);
        const valuesTensor = (valueNetwork.predict(input, { batchSize }) as tf.Tensor).squeeze();

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

        const normalizedAdvantages = normalize(advantages);

        return {
            advantages: normalizedAdvantages,
            tdErrors: tdErrors,
            returns: vTraces,
            values: values,
            logStdShift: -1 * (syncUnwrapTensor(phiPower.mean())[0]),
            // just for logs
            pureMean: syncUnwrapTensor(mean) as Float32Array,
            pureLogStd: syncUnwrapTensor(pureLogStdEff) as Float32Array,
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
    // bootstrap v̂_{T} = values[T]
    let vtp1 = dones[T - 1] ? 0 : values[T];
    if (vtp1 === undefined) { throw new Error('Implementation required last state as terminal'); }

    const vTraces = new Float32Array(T);
    const tdErrors = new Float32Array(T);
    const advantages = new Float32Array(T);

    // bootstrap: v̂_T = done? 0 : V(s_T)
    let nextAdv = 0; // A_{t+1}

    for (let t = T - 1; t >= 0; --t) {
        const nextValue = dones[t] ? 0 : values[t + 1];
        const discount = dones[t] ? 0 : gamma;
        const value = values[t];

        const c = dones[t] ? 0 : Math.min(rhos[t], clipC);
        const rho = Math.min(rhos[t], clipRho);
        const rhoPG = Math.min(rhos[t], clipRhoPG);

        // δ_t^V = r_t + γ V(s_{t+1}) - V(s_t)
        const tdError = rewards[t] + discount * nextValue - value;
        tdErrors[t] = tdError;

        // v̂_t = V(s_t) + ρ̄_t δ_t^V + γ c̄_t (v̂_{t+1} - V(s_{t+1}))
        vTraces[t] = value + rho * tdError + discount * c * (vtp1 - nextValue);

        // A_t = ρ̄_t^{PG} δ_t^V + γ c̄_t A_{t+1}
        const adv = rhoPG * tdError + discount * c * nextAdv;
        advantages[t] = adv;

        vtp1 = vTraces[t];
        nextAdv = adv;
    }

    return { vTraces, advantages, tdErrors };
}

function parsePolicyOutput(prediction: tf.Tensor | tf.Tensor[], minLogStd: number[], maxLogStd: number[]) {
    const [mean, logStd, phi] = prediction as [tf.Tensor2D, tf.Tensor2D, tf.Tensor2D];
    return { mean, logStd, phi }; // logStd: tanhMapToRange(logStd, minLogStd, maxLogStd)
}

function tanhMapToRange(z: tf.Tensor, min: number[], max: number[]) {
    const tMin = tf.tensor1d(min);
    const tMax = tf.tensor1d(max);
    const c = tMin.add(tMax).mul(0.5);
    const r = tMax.sub(tMin).mul(0.5);
    return tf.tanh(z).mul(r).add(c);
}

function computeEffectiveStd(logStd: tf.Tensor, phi: tf.Tensor) {
    const power = tf.log(phi.square().sum(1, true).add(1e-6)).mul(0.5); // [B, 1]
    const logStdEff = logStd.add(power); // [B, A]
    const stdEff = tf.exp(logStdEff);
    return { stdEff, power, pureLogStdEff: logStdEff };
}

export function softClipByValue(
    z: tf.Tensor,
    min: number,
    max: number,
): tf.Tensor {
    const normalized = z.sub(min).div(max - min); // [0, 1]
    const scaled = normalized.mul(2).sub(1); // [-1, 1]
    const clipped = tf.tanh(scaled.mul(2)).mul(0.5).add(0.5); // плавный клип
    return clipped.mul(max - min).add(min);
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

