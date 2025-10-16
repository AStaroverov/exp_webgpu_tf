import * as tf from '@tensorflow/tfjs';
import { Scalar } from '@tensorflow/tfjs-core/dist/tensor';
import { NamedTensor } from '@tensorflow/tfjs-core/dist/tensor_types';
import { normalize } from '../../../../lib/math.ts';
import { random } from '../../../../lib/random.ts';
import { computeLogProb } from '../../../ml-common/computeLogProb.ts';
import { ACTION_DIM } from '../../../ml-common/consts.ts';
import { flatTypedArray } from '../../../ml-common/flat.ts';
import { InputArrays, prepareRandomInputArrays } from '../../../ml-common/InputArrays.ts';
import { createInputTensors } from '../../../ml-common/InputTensors.ts';
import { AgentMemoryBatch } from '../../../ml-common/Memory.ts';
import { arrayHealthCheck, asyncUnwrapTensor, onReadyRead, syncUnwrapTensor } from '../../../ml-common/Tensor.ts';

let tC: undefined | tf.Tensor

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
    minLogStd: number,
    maxLogStd: number,
    returnCost: boolean,
): undefined | tf.Tensor {
    const C = (tC ??= tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E)));

    return tf.tidy(() => {
        return optimize(network.optimizer, () => {
            const predicted = network.predict(states, { batchSize }) as tf.Tensor[];
            const { mean, logStd } = parsePredict(predicted, minLogStd, maxLogStd);
            const std = logStd.exp();
            const newLogProbs = computeLogProb(actions, mean, std);
            const ratio = tf.exp(newLogProbs.sub(oldLogProbs));
            const clippedRatio = ratio.clipByValue(1 - clipRatio, 1 + clipRatio);
            const surr1 = ratio.mul(advantages);
            const surr2 = clippedRatio.mul(advantages);
            const policyLoss = tf.minimum(surr1, surr2).mean().mul(-1);
            const entropyLoss = logStd.add(C).sum(1).mean().mul(entropyCoeff);

            const totalLoss = policyLoss.sub(entropyLoss);
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
    batchSize: number,
    minLogStd: number,
    maxLogStd: number,
) {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, { batchSize }) as tf.Tensor[];
        const { mean, logStd } = parsePredict(predicted, minLogStd, maxLogStd);
        const std = logStd.exp();
        const newLogProbs = computeLogProb(actions, mean, std);
        const diff = oldLogProb.sub(newLogProbs);
        const kl = diff.mean().abs();
        return kl;
    });
}

export function computeKullbackLeiblerExact(
    policyNetwork: tf.LayersModel,
    states: tf.Tensor[],
    oldMean: tf.Tensor,
    oldLogStd: tf.Tensor,
    batchSize: number,
    minLogStd: number,
    maxLogStd: number,
): tf.Tensor {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(states, { batchSize }) as tf.Tensor[];
        const { mean: newMean, logStd: newLogStd } = parsePredict(predicted, minLogStd, maxLogStd);

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
    minLogStd: number,
    maxLogStd: number,
    noise?: tf.Tensor,
): {
    actions: Float32Array,
    mean: Float32Array,
    logStd: Float32Array,
    logProb: number
} {
    return tf.tidy(() => {
        const predicted = policyNetwork.predict(createInputTensors([state])) as tf.Tensor[];
        const { mean, logStd } = parsePredict(predicted, minLogStd, maxLogStd);
        const std = logStd.exp();

        const actions = noise ? mean.add(noise.mul(std)) : mean.clone();
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
    batchSize: number,
    gamma: number,
    minLogStd: number,
    maxLogStd: number,
    clipRho: number = 1.0,
    clipC: number = 1.0,
    clipRhoPG: number = 1.0,
): {
    advantages: Float32Array,
    tdErrors: Float32Array,
    returns: Float32Array,
    values: Float32Array,
    logStd: Float32Array,
} {
    return tf.tidy(() => {
        const input = createInputTensors(batch.states);
        const predicted = policyNetwork.predict(input, { batchSize }) as tf.Tensor[];
        const { mean: meanCurrent, logStd: logStdCurrent, pureLogStd: pureLogStdCurrent } = parsePredict(predicted, minLogStd, maxLogStd);
        const actions = tf.tensor2d(flatTypedArray(batch.actions), [batch.size, ACTION_DIM]);

        const logProbCurrentTensor = computeLogProb(actions, meanCurrent, logStdCurrent.exp());
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

        return {
            advantages: normalize(advantages),
            tdErrors: tdErrors,
            returns: vTraces,
            values: values,
            logStd: syncUnwrapTensor(pureLogStdCurrent) as Float32Array,
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
        const discount = dones[t] ? 0 : gamma;
        const nextValue = dones[t] ? 0 : values[t + 1];
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

function parsePredict(predict: tf.Tensor[], minLogStd: number, maxLogStd: number) {
    const outMean: tf.Tensor = predict[0];
    const outLogStd: tf.Tensor = predict[1];

    let clippedLogStd = outLogStd;

    if (isFinite(minLogStd) && isFinite(maxLogStd)) {
        // hard clipping
        // const clippedLogStd = outLogStd.clipByValue(minLogStd, maxLogStd);

        // sigmoid-based clipping
        // clippedLogStd = tf.add(minLogStd, tf.mul(maxLogStd - minLogStd, tf.sigmoid(outLogStd)));

        // tanh - based clipping
        // const c = (maxLogStd + minLogStd) / 2;
        // const r = (maxLogStd - minLogStd) / 2;
        // clippedLogStd = tf.add(c, tf.mul(r, tf.tanh(outLogStd)));

        // tanh-based with temperature
        // const tau = 2.0;
        // const span = maxLogStd - minLogStd;
        // const soft = tf.tanh(outLogStd.div(tau));         // (-1,1)
        // const s = soft.mul(0.5).add(0.5);
        // clippedLogStd = tf.add(minLogStd, tf.mul(span, s));

        // softsign-based clipping
        // const span = maxLogStd - minLogStd;
        // const softsign = outLogStd.div(tf.add(1, tf.abs(outLogStd)));
        // const s = softsign.div(2).add(0.5);   // в [0,1]
        // clippedLogStd = tf.add(minLogStd, tf.mul(span, s));

        // Hard softsign-based clipping
        const alpha = 3;  // >1 — делает насыщение быстрее (центральная зона)
        const p = 3;      // >=1 — чем больше, тем резче прижимает к краям
        const span = maxLogStd - minLogStd;
        // Жёсткий softsign: u / (1 + |u|^p), с предварительным усилением alpha
        // x = alpha * u
        const x = outLogStd.mul(alpha);
        // |x|^p
        const axp = tf.pow(tf.abs(x), tf.scalar(p));
        // soft_p = sign(x) * |x|^p / (1 + |x|^p)  -> в (-1, 1),  |x|->∞ => ±1
        const soft = tf.sign(x).mul(axp.div(tf.add(1, axp)));
        // Нормируем в [0,1]
        const s = soft.mul(0.5).add(0.5);
        // Маппим в [minLogStd, maxLogStd]
        clippedLogStd = tf.add(minLogStd, tf.mul(span, s));

        // const r = clippedLogStd.dataSync()
        // console.log([...outLogStd.dataSync()].map((v, i) => `${v.toFixed(3)} -> ${r[i].toFixed(6)}`));
        // debugger
    }

    return { mean: outMean, logStd: clippedLogStd, pureLogStd: outLogStd };
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
