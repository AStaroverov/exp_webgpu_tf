// Буфер опыта для PPO
import * as tf from '@tensorflow/tfjs';
import { shuffle } from '../../../../../lib/shuffle.ts';
import { abs, max, min } from '../../../../../lib/math.ts';
import { isDevtoolsOpen } from '../Common/utils.ts';

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

        return {
            size: this.states.length,
            states: tf.stack(this.states),
            actions: tf.stack(this.actions),
            logProbs: tf.stack(this.logProbs),
            values: tf.stack(this.values),
            rewards: tf.tensor1d(this.rewards),
            dones: tf.tensor1d(this.dones.map(done => done ? 1.0 : 0.0)),
            returns: tf.tensor1d(returns),
            advantages: tf.tensor1d(advantages),
        };
    }

    computeReturnsAndAdvantages(gamma: number, lam: number, lastValue: number = 0) {
        const n = this.states.length;
        const returns: number[] = new Array(n).fill(0);
        const advantages: number[] = new Array(n).fill(0);

        const rewards = this.rewards; //.map(r => signedLog(r, 10));
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
            const delta = rewards[i]
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

        if (isDevtoolsOpen()) {
            const minRew = min(...rewards);
            const maxRew = max(...rewards);
            const minVal = min(...valuesArr);
            const maxVal = max(...valuesArr);
            const minRet = min(...returns);
            const maxRet = max(...returns);
            const minAdv = min(...normalizedAdvantages);
            const maxAdv = max(...normalizedAdvantages);

            console.log('[R&A]'
                , '  Rew:', strgify(minRew), strgify(maxRew)
                , '| Val:', strgify(minVal), strgify(maxVal)
                , '| Ret:', strgify(minRet), strgify(maxRet)
                , '| Adv:', strgify(minAdv), strgify(maxAdv),
            );
        }

        return {
            returns: returns,
            advantages: normalizedAdvantages,  // Возвращаем нормализованные advantages
        };
    }

    dispose() {
        // Освобождаем все тензоры
        this.states.forEach(state => state.dispose());
        this.actions.forEach(action => action.dispose());
        this.logProbs.forEach(logProb => logProb.dispose());
        this.values.forEach(value => value.dispose());

        // Сбрасываем массивы
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.values = [];
        this.rewards = [];
        this.dones = [];
        this.tmpRewards = [];
        this.tmpDones = [];
    }

    private collapseTmpData() {
        this.rewards.push(this.tmpRewards.reduce((acc, val) => acc + val, 0));
        this.dones.push(this.tmpDones.reduce((acc, val) => acc && val, true));
        this.tmpRewards = [];
        this.tmpDones = [];
    }
}

function strgify(v: number): string {
    return (v > 0 ? ' ' : '-') + abs(v).toFixed(2);
}