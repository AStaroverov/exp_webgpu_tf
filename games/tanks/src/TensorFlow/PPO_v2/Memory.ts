// Буфер опыта для PPO
import * as tf from '@tensorflow/tfjs';
import { shuffle } from '../../../../../lib/shuffle.ts';
import { max, min } from '../../../../../lib/math.ts';

export type Batch = {
    size: number,
    states: tf.Tensor,
    actions: tf.Tensor,
    logProbs: tf.Tensor,
    values: tf.Tensor,
    rewards: tf.Tensor,
    dones: tf.Tensor,
    returns: tf.Tensor,
    advantages: tf.Tensor,
}

export class Memory {
    private map = new Map<number, SubMemory>();

    constructor() {
    }

    size() {
        let size = 0;

        this.map.forEach(subMemory => {
            size += subMemory.size();
        });

        return size;
    }

    addFirstPart(id: number, state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.addFirstPart(state, action, logProb, value);
    }

    updateSecondPart(id: number, reward: number, done: boolean, isLast = false) {
        if (!this.map.has(id)) {
            throw new Error('SubMemory not found');
        }
        this.map.get(id)!.updateSecondPart(reward, done, isLast);
    }

    getBatch(gamma: number, lam: number): Batch {
        const batches = this.getBatches(gamma, lam);
        const values = shuffle(Array.from(batches.values()));

        return {
            size: values.reduce((acc, batch) => acc + batch.size, 0),
            states: tf.concat(values.map(batch => batch.states)),
            actions: tf.concat(values.map(batch => batch.actions)),
            logProbs: tf.concat(values.map(batch => batch.logProbs)),
            values: tf.concat(values.map(batch => batch.values)),
            rewards: tf.concat(values.map(batch => batch.rewards)),
            dones: tf.concat(values.map(batch => batch.dones)),
            returns: tf.concat(values.map(batch => batch.returns)),
            advantages: tf.concat(values.map(batch => batch.advantages)),
        };
    }

    // Метод для получения батча для обучения
    getBatches(gamma: number, lam: number) {
        const batches = new Map<number, Batch>();

        this.map.forEach((subMemory, id) => {
            batches.set(id, subMemory.getBatch(gamma, lam));
        });

        return batches;
    }

    dispose() {
        this.map.forEach(subMemory => subMemory.dispose());
        this.map.clear();
    }
}

export class SubMemory {
    private states: tf.Tensor[] = [];
    private actions: tf.Tensor[] = [];
    private logProbs: tf.Tensor[] = [];
    private values: tf.Tensor[] = [];
    private rewards: number[] = [];
    private dones: boolean[] = [];

    private tmpRewards: number[] = [];
    private tmpDones: boolean[] = [];

    private returns?: tf.Tensor;
    private advantages?: tf.Tensor;

    constructor() {
    }

    size() {
        return this.states.length;
    }

    addFirstPart(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor) {
        this.states.push(state);
        this.actions.push(action);
        this.logProbs.push(logProb);
        this.values.push(value);
    }

    updateSecondPart(reward: number, done: boolean, isLast = false) {
        this.tmpRewards.push(reward);
        this.tmpDones.push(done);

        if (isLast) {
            this.collapseTmpData();
        }
    }

    // Метод для получения батча для обучения
    getBatch(gamma: number, lam: number): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
        }
        if (this.tmpDones.length > 0 || this.tmpRewards.length > 0) {
            this.collapseTmpData();
        }
        if (this.states.length !== this.rewards.length || this.states.length !== this.dones.length) {
            const minLen = Math.min(this.states.length, this.rewards.length, this.dones.length);
            this.states.length = minLen;
            this.actions.length = minLen;
            this.logProbs.length = minLen;
            this.values.length = minLen;
            this.rewards.length = minLen;
            this.dones.length = minLen;
        }

        const { returns, advantages } = this.computeReturnsAndAdvantages(gamma, lam);
        this.returns = returns;
        this.advantages = advantages;

        return {
            size: this.states.length,
            states: tf.stack(this.states),
            actions: tf.stack(this.actions),
            logProbs: tf.stack(this.logProbs),
            values: tf.stack(this.values),
            rewards: tf.tensor1d(this.rewards),
            dones: tf.tensor1d(this.dones.map(done => done ? 1.0 : 0.0)),
            advantages,
            returns,
        };
    }

    computeReturnsAndAdvantages(gamma: number, lam: number, lastValue: number = 0) {
        const n = this.states.length;
        const returns: number[] = new Array(n).fill(0);
        const advantages: number[] = new Array(n).fill(0);

        // Скачаем values в CPU, чтоб проще считать
        const valuesArr = tf.stack(this.values).dataSync(); // shape [n]

        let adv = 0;
        // bootstrap, если последний transition не done
        let lastVal = lastValue; // Если done в конце, возьмём 0

        // Идём с конца вперёд
        for (let i = n - 1; i >= 0; i--) {
            if (this.dones[i]) {
                // если done, то обнуляем хвост
                adv = 0;
                lastVal = 0;
            }
            const delta = this.rewards[i]
                + gamma * lastVal * (this.dones[i] ? 0 : 1)
                - valuesArr[i];
            adv = delta + gamma * lam * adv * (this.dones[i] ? 0 : 1);

            advantages[i] = adv;
            returns[i] = valuesArr[i] + adv;

            lastVal = valuesArr[i];
        }

        // Нормализация advantages
        const advMean = advantages.reduce((sum, val) => sum + val, 0) / n;
        const advStd = Math.sqrt(
            advantages.reduce((sum, val) => sum + Math.pow(val - advMean, 2), 0) / n,
        );
        const normalizedAdvantages = advantages.map(adv => (adv - advMean) / (advStd + 1e-8));
        // const normalizedAdvantages = advantages.map(adv => signedLog(adv, 1));
        // const normalizedAdvantages = linearScale(signedLogAdvantages, -3, 3);

        const minRet = min(...returns);
        const maxRet = max(...returns);
        const orgMinAdv = min(...advantages);
        const orgMaxAdv = max(...advantages);
        const minAdv = min(...normalizedAdvantages);
        const maxAdv = max(...normalizedAdvantages);
        const negativeAdvSum = normalizedAdvantages.reduce((sum, val) => sum + Math.min(val, 0), 0);
        const positiveAdvSum = normalizedAdvantages.reduce((sum, val) => sum + Math.max(val, 0), 0);

        console.log('[Returns] Min/Max:', minRet.toFixed(2), maxRet.toFixed(2));
        console.log('[Advantages]: Original Min/Max', orgMinAdv.toFixed(2), orgMaxAdv.toFixed(2));
        console.log('[Advantages]: Normaliz Min/Max', minAdv.toFixed(2), maxAdv.toFixed(2));
        console.log('[Advantages]: Negative/Positive Sum', negativeAdvSum.toFixed(2), positiveAdvSum.toFixed(2));

        return {
            returns: tf.tensor1d(returns),
            advantages: tf.tensor1d(normalizedAdvantages),  // Возвращаем нормализованные advantages
        };
    }

    dispose() {
        // Освобождаем все тензоры
        this.states.forEach(state => state.dispose());
        this.actions.forEach(action => action.dispose());
        this.logProbs.forEach(logProb => logProb.dispose());
        this.values.forEach(value => value.dispose());
        this.advantages?.dispose();
        this.returns?.dispose();

        // Сбрасываем массивы
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.values = [];
        this.rewards = [];
        this.dones = [];
    }

    private collapseTmpData() {
        this.rewards.push(this.tmpRewards.reduce((acc, val) => acc + val, 0));
        this.dones.push(this.tmpDones.reduce((acc, val) => acc && val, true));
        this.tmpRewards = [];
        this.tmpDones = [];
    }
}
