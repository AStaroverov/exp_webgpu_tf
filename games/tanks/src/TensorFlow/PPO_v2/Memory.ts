// Буфер опыта для PPO
import * as tf from '@tensorflow/tfjs';
import { shuffle } from '../../../../../lib/shuffle.ts';

type Batch = {
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

    add(id: number, state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor, reward: number, done: boolean) {
        if (!this.map.has(id)) {
            this.map.set(id, new SubMemory());
        }
        this.map.get(id)!.add(state, action, logProb, value, reward, done);
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

    private returns?: tf.Tensor;
    private advantages?: tf.Tensor;

    constructor() {
    }

    size() {
        return this.states.length;
    }

    add(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, value: tf.Tensor, reward: number, done: boolean) {
        this.states.push(state.clone());
        this.actions.push(action.clone());
        this.logProbs.push(logProb.clone());
        this.values.push(value.clone());
        this.rewards.push(reward);
        this.dones.push(done);
    }

    // Метод для получения батча для обучения
    getBatch(gamma: number, lam: number): Batch {
        if (this.states.length === 0) {
            throw new Error('Memory is empty');
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

        const minAdv = Math.min(...advantages);
        const maxAdv = Math.max(...advantages);
        const minRet = Math.min(...returns);
        const maxRet = Math.max(...returns);
        console.log('Min/Max advantages:', minAdv, maxAdv);
        console.log('Min/Max returns:', minRet, maxRet);

        return { returns: tf.tensor1d(returns), advantages: tf.tensor1d(advantages) };
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
}