// Буфер опыта для PPO
import { shuffle } from '../../../../../lib/shuffle.ts';
import { abs, max, min } from '../../../../../lib/math.ts';
import { isDevtoolsOpen } from './uiUtils.ts';
import { InputArrays } from './InputArrays.ts';

export type Batch = {
    states: InputArrays[],
    actions: Float32Array[],
    logProbs: number[],
    values: number[],
    returns: number[],
    advantages: number[],
    // meta
    size: number,
    dones: number[],
    rewards: number[],
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

    toArray() {
        return Array.from(this.map.values());
    }

    addFirstPart(id: number, state: InputArrays, action: Float32Array, logProb: number, value: number) {
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
        const batch = {
            size: values.reduce((acc, batch) => acc + batch.size, 0),
            states: (values.map(batch => batch.states)).flat(),
            actions: (values.map(batch => batch.actions)).flat(),
            logProbs: (values.map(batch => batch.logProbs)).flat(),
            values: (values.map(batch => batch.values)).flat(),
            rewards: (values.map(batch => batch.rewards)).flat(),
            dones: (values.map(batch => batch.dones)).flat(),
            returns: (values.map(batch => batch.returns)).flat(),
            advantages: (values.map(batch => batch.advantages)).flat(),
        };

        return batch;
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
    private states: InputArrays[] = [];
    private actions: Float32Array[] = [];
    private logProbs: number[] = [];
    private values: number[] = [];
    private rewards: number[] = [];
    private dones: boolean[] = [];

    private tmpRewards: number[] = [];
    private tmpDones: boolean[] = [];

    constructor() {
    }

    size() {
        return this.states.length;
    }

    addFirstPart(state: InputArrays, action: Float32Array, logProb: number, value: number) {
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
            states: (this.states),
            actions: (this.actions),
            logProbs: (this.logProbs),
            values: (this.values),
            rewards: (this.rewards),
            dones: (this.dones.map(done => done ? 1.0 : 0.0)),
            returns: (returns),
            advantages: (advantages),
        };
    }

    computeReturnsAndAdvantages(gamma: number, lam: number, lastValue: number = 0) {
        const n = this.states.length;
        const returns: number[] = new Array(n).fill(0);
        const advantages: number[] = new Array(n).fill(0);

        const rewards = this.rewards; //.map(r => signedLog(r, 10));
        const values = this.values; // shape [n]

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
                - values[i];
            adv = delta + gamma * lam * adv * (this.dones[i] ? 0 : 1);

            advantages[i] = adv;
            returns[i] = values[i] + adv;

            lastVal = values[i];
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
            const minVal = min(...values);
            const maxVal = max(...values);
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