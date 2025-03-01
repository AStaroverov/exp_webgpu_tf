import * as tf from '@tensorflow/tfjs';

const GAMMA = 0.99; // Discount factor
const LAMBDA = 0.95; // GAE parameter

export class PrioritizedExperienceBuffer {
    states: tf.Tensor[] = [];
    actions: tf.Tensor[] = [];
    oldLogProbs: tf.Tensor[] = [];
    rewards: number[] = [];
    values: tf.Tensor[] = [];
    dones: boolean[] = []; // Добавлено для отметки терминальных состояний
    priorities: number[] = []; // Приоритеты для выборки
    alpha: number = 0.6; // Показатель степени для приоритета (контролирует степень приоритизации)
    beta: number = 0.4; // Исходное значение бета для важности выборки (начинаем с 0.4 и увеличиваем до 1)
    betaAnnealing: number = 0.001; // Скорость увеличения бета

    constructor(private capacity: number = 1024 * 2) {
    }

    get size() {
        return this.states.length;
    }

    add(state: tf.Tensor, action: tf.Tensor, logProb: tf.Tensor, reward: number, value: tf.Tensor, done: boolean, priority: number = 1.0) {
        this.states.push(state);
        this.actions.push(action);
        this.oldLogProbs.push(logProb);
        this.rewards.push(reward);
        this.values.push(value);
        this.dones.push(done);
        this.priorities.push(priority);

        // Trim if exceeding capacity
        if (this.states.length > this.capacity) {
            // Clean up tensors before removing
            this.states[0].dispose();
            this.actions[0].dispose();
            this.oldLogProbs[0].dispose();
            this.values[0].dispose();

            this.states.shift();
            this.actions.shift();
            this.oldLogProbs.shift();
            this.rewards.shift();
            this.values.shift();
            this.dones.shift();
            this.priorities.shift();
        }
    }

    clear() {
        // Clean up tensors
        this.states.forEach(t => t.dispose());
        this.actions.forEach(t => t.dispose());
        this.oldLogProbs.forEach(t => t.dispose());
        this.values.forEach(t => t.dispose());

        this.states = [];
        this.actions = [];
        this.oldLogProbs = [];
        this.rewards = [];
        this.values = [];
        this.dones = [];
        this.priorities = [];
    }

    // Update priorities for experiences (typically after computing TD errors)
    updatePriorities(indices: number[], newPriorities: number[]) {
        for (let i = 0; i < indices.length; i++) {
            if (indices[i] < this.priorities.length) {
                this.priorities[indices[i]] = Math.max(0.00001, newPriorities[i]);
            }
        }
    }

    // Calculate advantages using Generalized Advantage Estimation (GAE)
    computeReturnsAndAdvantages(gamma = GAMMA, lambda = LAMBDA): [tf.Tensor, tf.Tensor] {
        const returns: number[] = new Array(this.rewards.length);
        const advantages: number[] = new Array(this.rewards.length);

        let nextReturn = 0;
        let nextAdvantage = 0;

        // Первый проход: вычисление исходных значений
        for (let i = this.rewards.length - 1; i >= 0; i--) {
            const reward = this.rewards[i];
            const value = this.values[i].dataSync()[0];
            const done = this.dones[i];

            const nextStateValue = done ? 0 : (i < this.rewards.length - 1 ? this.values[i + 1].dataSync()[0] : 0);

            // Клиппинг разницы между наградой и ценностью для избежания экстремальных значений
            const delta = Math.max(-50, Math.min(50, reward + (done ? 0 : gamma * nextStateValue) - value));

            // Аккумулируем преимущество с ограничением
            nextAdvantage = delta + (done ? 0 : gamma * lambda * nextAdvantage);
            nextAdvantage = Math.max(-100, Math.min(100, nextAdvantage)); // Явное ограничение

            // Аккумулируем возвращаемое значение с ограничением
            nextReturn = reward + (done ? 0 : gamma * nextReturn);
            nextReturn = Math.max(-100, Math.min(100, nextReturn)); // Явное ограничение

            returns[i] = nextReturn;
            advantages[i] = nextAdvantage;
        }

        // Второй проход: нормализация и более агрессивное клиппинг
        let advantageSum = 0;
        for (let i = 0; i < advantages.length; i++) {
            advantageSum += advantages[i];
        }
        const advantageMean = advantageSum / advantages.length;

        let advantageSqSum = 0;
        for (let i = 0; i < advantages.length; i++) {
            advantageSqSum += Math.pow(advantages[i] - advantageMean, 2);
        }
        const advantageStd = Math.sqrt(advantageSqSum / advantages.length) + 1e-8;

        // Нормализация с жёстким клиппингом до диапазона [-5, 5]
        const normalizedAdvantages = advantages.map(adv => {
            const normalized = (adv - advantageMean) / advantageStd;
            return Math.max(-5, Math.min(5, normalized)); // Жесткий клиппинг
        });

        // Аналогично для returns
        let returnSum = 0;
        for (let i = 0; i < returns.length; i++) {
            returnSum += returns[i];
        }
        const returnMean = returnSum / returns.length;

        let returnSqSum = 0;
        for (let i = 0; i < returns.length; i++) {
            returnSqSum += Math.pow(returns[i] - returnMean, 2);
        }
        const returnStd = Math.sqrt(returnSqSum / returns.length) + 1e-8;

        const normalizedReturns = returns.map(ret => {
            const normalized = (ret - returnMean) / returnStd;
            return Math.max(-5, Math.min(5, normalized)); // Жесткий клиппинг
        });

        // Проверка на NaN
        if (normalizedReturns.some(isNaN)) {
            console.error('NaN in normalized returns:', returns, returnMean, returnStd);
        }
        if (normalizedAdvantages.some(isNaN)) {
            console.error('NaN in normalized advantages:', advantages, advantageMean, advantageStd);
        }

        // Необязательно: лог статистики для мониторинга
        const advMax = Math.max(...normalizedAdvantages);
        const advMin = Math.min(...normalizedAdvantages);
        const retMax = Math.max(...normalizedReturns);
        const retMin = Math.min(...normalizedReturns);
        console.log(`Advantage range: [${ advMin.toFixed(2) }, ${ advMax.toFixed(2) }], Return range: [${ retMin.toFixed(2) }, ${ retMax.toFixed(2) }]`);

        return [
            tf.tensor1d(normalizedReturns),
            tf.tensor1d(normalizedAdvantages),
        ];
    }

    // Выборка с приоритезацией
    getBatch(batchSize: number): [tf.Tensor[], tf.Tensor[], tf.Tensor[], tf.Tensor, tf.Tensor, number[]] {
        if (this.size < batchSize) {
            throw new Error(`Buffer size (${ this.size }) smaller than requested batch size (${ batchSize })`);
        }

        // Анализ бета для корректировки важности выборки
        this.beta = Math.min(1.0, this.beta + this.betaAnnealing);

        // Перевести приоритеты в вероятности выборки
        const prioritySum = this.priorities.reduce((a, b) => a + Math.pow(b, this.alpha), 0);
        const probabilities = this.priorities.map(p => Math.pow(p, this.alpha) / prioritySum);

        // Выборка индексов на основе вероятностей
        const indices: number[] = [];
        for (let i = 0; i < batchSize; i++) {
            let idx = 0;
            let r = Math.random();
            let sum = 0;
            for (let j = 0; j < probabilities.length; j++) {
                sum += probabilities[j];
                if (r <= sum) {
                    idx = j;
                    break;
                }
            }
            indices.push(idx);
        }

        // Рассчитать веса важности выборки для корректировки смещения
        const maxWeight = Math.pow(this.size * Math.min(...probabilities), -this.beta);
        const weights = indices.map(i => Math.pow(this.size * probabilities[i], -this.beta) / maxWeight);

        // Get batch elements
        const batchStates = indices.map(i => this.states[i]);
        const batchActions = indices.map(i => this.actions[i]);
        const batchLogProbs = indices.map(i => this.oldLogProbs[i]);

        // Calculate returns and advantages
        const [returns, advantages] = this.computeReturnsAndAdvantages();

        const batchReturns = tf.gather(returns, indices);
        const batchAdvantages = tf.gather(advantages, indices);

        // Normalize advantages
        const mean = tf.mean(batchAdvantages);
        const std = tf.sqrt(tf.mean(tf.square(tf.sub(batchAdvantages, mean))));
        const normalizedAdvantages = tf.div(tf.sub(batchAdvantages, mean), tf.add(std, 1e-8));

        // Умножить преимущества на веса важности
        const weightedAdvantages = tf.mul(normalizedAdvantages, tf.tensor1d(weights));

        returns.dispose();
        advantages.dispose();
        mean.dispose();
        std.dispose();
        normalizedAdvantages.dispose();

        return [batchStates, batchActions, batchLogProbs, batchReturns, weightedAdvantages, indices];
    }
}